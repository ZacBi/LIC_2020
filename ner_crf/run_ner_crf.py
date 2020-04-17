# pylint: disable=bad-continuation

import glob
import logging
import os
import json

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
    TensorDataset,
)

from transformers import (
    WEIGHTS_NAME,
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
    AdamW,
    BertConfig,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm, trange

from ner_crf.utils import (
    json_to_text,
    seed_everything,
    get_label_map_by_file,
)

from ner_crf.models import BertCRF
from ner_crf.metrics import SeqEntityScore
from ner_crf.args import get_args
from ner_crf.log import logging_train, logging_continuing_training
from ner_crf.processors import (
    CluenerProcessor,
    collate_fn,
    convert_examples_to_features,
)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys())
                  for conf in MODEL_CONFIG_CLASSES), ())

MODEL_CLASSES = {
    ## bert ernie bert_wwm bert_wwwm_ext,roberta
    'bert': (BertConfig, BertCRF, BertTokenizer)
}

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
processor = CluenerProcessor()  # pylint: disable=invalid-name


def prepare_optimizer_and_scheduler(args, model, t_total):
    """Prepare optimizer and schedule (linear warmup and decay)

    Args:
        t_total: total train steps
        model: model
    """
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

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        eps=args.adam_epsilon,
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=t_total)

    return optimizer, scheduler


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        collate_fn=collate_fn,
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps \
            // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) \
                // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    optimizer, scheduler = prepare_optimizer_and_scheduler(
        args, model, t_total)

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) \
            and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(
            torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(
            torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    # Set for fp16 training
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("No module apex")
        model, optimizer = amp.initialize(model,
                                          optimizer,
                                          opt_level=args.fp16_opt_level)

    # Logging for training!
    logging_train(args, len(train_dataset), t_total)
    # NOTE: Check if continuing training from a checkpoint
    logging_continuing_training(args, len(train_dataloader))

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    # NOTE: begin training
    model.zero_grad()
    seed_everything(args.seed)
    tr_loss, logging_loss = 0.0, 0.0
    train_iterator = trange(epochs_trained,
                            int(args.num_train_epochs),
                            desc="Epoch")
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc='Training')
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            # NOTE: add input_lens for every examples
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3],
            }
            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet"] else None)

            # NOTE: model outputs are always tuple in transformers (see doc)
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

                # Update learning rate schedule
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                # NOTE: logging train loss for **every batch**, notice that we use
                # gradient accumulation here.
                tb_writer.add_scalar("Train/lr",
                                     scheduler.get_lr()[0], global_step)
                tb_writer.add_scalar("Train/loss", tr_loss - logging_loss,
                                     global_step)
                logging_loss = tr_loss

                # NOTE: Evaluate while training
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    results = evaluate(args, model, tokenizer)
                    for key, value in results.items():
                        tb_writer.add_scalar("Eval/{}".format(key), value,
                                             global_step)

                # NOTE: Save model
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir,
                                              "ckpt-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    # Take care of distributed/parallel training
                    model_to_save = (model.module
                                     if hasattr(model, "module") else model)
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

    # End Training
    tb_writer.close()
    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix=""):
    metric = SeqEntityScore(args.id2label)

    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)
    eval_dataset = load_and_cache_examples(args, tokenizer, mode='dev')

    args.eval_batch_size = args.per_gpu_eval_batch_size
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size,
        collate_fn=collate_fn,
    )

    # NOTE: Eval!
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0

    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluation"):
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3],
            }

            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet"] else None)

            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            tags, _ = model.crf._viterbi_decode(logits,
                                                inputs['attention_mask'])
            eval_loss += tmp_eval_loss.item()

        nb_eval_steps += 1
        batch_size, _ = inputs['input_ids'].shape
        out_label_ids = inputs['labels'].detach().cpu().tolist()
        last_valid_idx = inputs['attention_mask'].int().sum(1) - 1
        # FIXME: maybe error will be thrown!!!
        for batch_idx in range(batch_size):
            cls_idx, sep_idx = 0, last_valid_idx[batch_idx].item()
            pred_label_ids = tags[batch_idx]
            true_label_ids = out_label_ids[batch_idx][cls_idx + 1:sep_idx]

            assert len(pred_label_ids) == len(true_label_ids)
            metric.update(y_true=true_label_ids, y_pred=pred_label_ids)

    eval_loss = eval_loss / nb_eval_steps
    results = metric.get_result()
    results['loss'] = eval_loss
    logger.info("***** Eval results %s *****", prefix)
    info = "-".join(
        [f' {key}: {value:.4f} ' for key, value in results.items()])
    logger.info(info)
    return results


def predict(args, model, tokenizer, prefix=""):
    pred_output_dir = args.output_dir
    if not os.path.exists(pred_output_dir):
        os.makedirs(pred_output_dir)
    test_dataset = load_and_cache_examples(args, tokenizer, mode='test')
    # Note that DistributedSampler samples randomly
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset,
                                 sampler=test_sampler,
                                 batch_size=1,
                                 collate_fn=collate_fn)
    # Eval!
    logger.info("***** Running prediction %s *****", prefix)
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", 1)

    results = []
    test_iterator = tqdm(test_dataloader, desc='Inference')
    for step, batch in enumerate(test_iterator):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": None,
            }
            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet"] else None)
            outputs = model(**inputs)
            logits = outputs[0]
            preds, _ = model.crf._viterbi_decode(logits,
                                                 inputs['attention_mask'])

        preds = preds[0][1:-1]  # [CLS]XXXX[SEP]
        label_entities = SeqEntityScore.get_entities(args.id2label, preds)
        json_d = {}
        json_d['id'] = step
        json_d['tag_seq'] = " ".join(preds)
        json_d['entities'] = label_entities
        results.append(json_d)

    output_predict_file = os.path.join(pred_output_dir, prefix,
                                       "test_prediction.json")
    output_submit_file = os.path.join(pred_output_dir, prefix,
                                      "test_submit.json")
    with open(output_predict_file, "w") as writer:
        for record in results:
            writer.write(json.dumps(record) + '\n')
    test_text = []
    with open(os.path.join(args.data_dir, "test.json"), 'r') as f_obj:
        for line in f_obj:
            test_text.append(json.loads(line))
    test_submit = []
    for truth, res in zip(test_text, results):
        json_d = {}
        json_d['id'] = truth['id']
        json_d['label'] = {}
        entities = res['entities']
        words = list(truth['text'])
        if entities:
            for subject in entities:
                tag = subject[0]
                start = subject[1]
                end = subject[2]
                word = "".join(words[start:end + 1])
                if tag in json_d['label']:
                    if word in json_d['label'][tag]:
                        json_d['label'][tag][word].append([start, end])
                    else:
                        json_d['label'][tag][word] = [[start, end]]
                else:
                    json_d['label'][tag] = {}
                    json_d['label'][tag][word] = [[start, end]]
        test_submit.append(json_d)
    json_to_text(output_submit_file, test_submit)


def load_and_cache_examples(args, tokenizer, mode='train'):
    # pylint: disable=not-callable
    # Load data features from cache or dataset file
    cached_file_name = 'cached_crf-{}_{}_{}'.format(
        mode,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.train_max_seq_length if mode ==
            'train' else args.eval_max_seq_length))
    cached_features_file = os.path.join(args.data_dir, cached_file_name)

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s",
                    cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode == 'train':
            examples = processor.get_train_examples(args.data_dir)
        if mode == 'dev':
            examples = processor.get_dev_examples(args.data_dir)
        if mode == 'test':
            examples = processor.get_test_examples(args.data_dir)

        features = convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            label2id=args.label2id,
            max_seq_length=args.train_max_seq_length if mode == 'train' \
                    else args.eval_max_seq_length,
        )

        logger.info("Saving features into cached file %s",
                    cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features],
                                 dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features],
                                  dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features],
                                      dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features],
                                 dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_token_type_ids,
                            all_label_ids)
    return dataset


def main():
    args = get_args()

    # Check output dir
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if os.path.exists(args.output_dir) \
            and os.listdir(args.output_dir) \
            and args.do_train \
            and not args.overwrite_output_dir:
        raise ValueError(
            f"Output directory ({args.output_dir}) already exists and is not empty. "
            "Use --overwrite_output_dir to overcome.")

    # Set device
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.device = device

    logger.warning("Process device: %s, 16-bits training: %s", device,
                   args.fp16)

    # Set seed
    seed_everything(args.seed)

    # Prepare NER task
    args.label2id = get_label_map_by_file(args.label_map_config)
    args.label2id = args.label2id
    args.id2label = {i: label for label, i in args.label2id.item()}
    args.num_labels = len(args.label2id)

    # Load pretrained model and tokenizer
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_labels,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name
        if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
        label2id=args.label2id,
        device=args.device)

    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)
    # NOTE: Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, mode='train')
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step,
                    tr_loss)

    # Saving best-practices: if you use defaults names for the model,
    # you can reload it using from_pretrained()
    if args.do_train:
        # Create output directory if needed
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (model.module if hasattr(model, "module") else model
                         )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_vocabulary(args.output_dir)
        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    # Evaluation
    results = {}
    if args.do_eval:
        tokenizer = tokenizer_class.from_pretrained(
            args.output_dir, do_lower_case=args.do_lower_case)

        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(
                    glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME,
                              recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(
                logging.WARN)  # Reduce logging

        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split(
                "-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split(
                '/')[-1] if checkpoint.find('ckpt') != -1 else ""
            model = model_class.from_pretrained(checkpoint,
                                                config=config,
                                                label2id=args.label2id,
                                                device=args.device)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix)
            if global_step:
                result = {
                    "{}_{}".format(global_step, k): v
                    for k, v in result.items()
                }
            results.update(result)
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("{} = {}\n".format(key, str(results[key])))

    if args.do_predict:
        tokenizer = tokenizer_class.from_pretrained(
            args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.predict_checkpoints > 0:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(
                    glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME,
                              recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(
                logging.WARN)  # Reduce logging
            checkpoints = [
                ckpt for ckpt in checkpoints
                if ckpt.split('-')[-1] == str(args.predict_checkpoints)
            ]
        logger.info("Predict the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            prefix = checkpoint.split(
                '/')[-1] if checkpoint.find('ckpt') != -1 else ""
            model = model_class.from_pretrained(checkpoint,
                                                config=config,
                                                label2id=args.label2id,
                                                device=args.device)
            model.to(args.device)
            predict(args, model, tokenizer, prefix=prefix)


if __name__ == "__main__":
    main()
