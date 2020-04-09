# pylint: disable=bad-continuation

import os
import random
import logging
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler

# transformers
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm, trange

from ner_crf.model.evaluate import (
    evaluate,
    load_and_cache_examples,
)

logger = logging.getLogger(__file__)  # pylint: disable=invalid-name


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def fp16_train(args, model, optimizer):
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )
        model, optimizer = amp.initialize(model,
                                          optimizer,
                                          opt_level=args.fp16_opt_level)


def continue_train_from_ckpt(args, nums_batch):
    "Continue trainning from checkpoint, if there is no checkpoint, train from scratch"

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    if os.path.exists(args.model_name_or_path):
        # set global_step to gobal_step of last saved checkpoint from model path
        try:
            global_step = int(
                args.model_name_or_path.split("-")[-1].split("/")[0])
        except ValueError:
            global_step = 0
        epochs_trained = global_step // (len(nums_batch) //
                                         args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (
            len(nums_batch) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, "
                    "will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch",
                    steps_trained_in_current_epoch)

    return global_step, epochs_trained, steps_trained_in_current_epoch


def train(
    args,
    train_dataset,
    model,
    tokenizer,
    labels,
    pad_token_label_id,
):
    """ Train the model """

    tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (
            len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(
            train_dataloader
        ) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay":
            args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay":
            0.0
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate,
                      eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=t_total)

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(
            args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
                os.path.join(args.model_name_or_path, "scheduler.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(
            torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(
            torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    # NOTE: Train in fp16 mode if args.fp16 is true
    fp16_train(args, model, optimizer)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size * args.gradient_accumulation_steps,
    )
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    # Check if continuing training from a checkpoint
    global_step, epochs_trained, steps_trained_in_current_epoch = continue_train_from_ckpt(
        args, len(train_dataloader))

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained,
        int(args.num_train_epochs),
        desc="Epoch",
        disable=False,
    )

    set_seed(args)  # Added here for reproductibility

    for _ in train_iterator:
        epoch_iterator = tqdm(
            train_dataloader,
            desc="Iteration",
            disable=False,
        )
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3]
            }
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet"] else None
                )  # XLM and RoBERTa don"t use segment_ids

            # model outputs are always tuple in pytorch-transformers (see doc)
            outputs = model(**inputs)
            loss = outputs[0]

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.logging_steps > 0 \
                        and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.evaluate_during_training:
                        results, _ = evaluate(
                            args,
                            model,
                            tokenizer,
                            labels,
                            pad_token_label_id,
                            mode="dev"
                        )
                        for key, value in results.items():
                            tb_writer.add_scalar("Eval/{}".format(key), value,
                                                 global_step)
                    tb_writer.add_scalar("Train/lr",
                                         scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("Train/loss",
                                         (tr_loss - logging_loss) /
                                         args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(
                        args.output_dir, "ckpt_{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args,
                               os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(),
                               os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(),
                               os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s",
                                output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    tb_writer.close()

    return global_step, tr_loss / global_step


def train_pipeline(args, labels, pad_token_label_id, tokenizer_args):
    """Train pipeline

    Args:
        args: args about train/evaluate/infer.
        labels: labels.
        pad_token_label_id: pad token label id for not caculate CrossEntropy.
        tokenizer_args: tokenizer args.
    """
    id2label = {str(i): label for i, label in enumerate(labels)}
    label2id = {label: i for i, label in enumerate(labels)}
    cache_dir = args.cache_dir if args.cache_dir else None
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        cache_dir=cache_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name
        if args.tokenizer_name else args.model_name_or_path,
        cache_dir=cache_dir,
        **tokenizer_args,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=cache_dir,
    )

    model.to(args.device)

    train_dataset = load_and_cache_examples(args,
                                            tokenizer,
                                            labels,
                                            pad_token_label_id,
                                            mode="train")

    # NOTE: train entry point
    global_step, tr_loss = train(args, train_dataset, model, tokenizer, labels,
                                 pad_token_label_id)
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model,
    # you can reload it using from_pretrained()
    # Create output directory if needed
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logger.info("Saving model checkpoint to %s", args.output_dir)
    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    # Take care of distributed/parallel training
    model_to_save = (model.module if hasattr(model, "module") else model)
    model_to_save.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Good practice: save your training arguments together with the trained model
    torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
