# coding=utf-8
# pylint: disable=bad-continuation

import os
import glob
import random
import logging

# torch group
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from seqeval.metrics import f1_score, precision_score, recall_score

# transformers
from transformers import (
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

import numpy as np
from tqdm import tqdm, trange

# Self package
from ner_crf.model.utils_ner import (
    get_labels,
    read_examples_from_file,
    convert_examples_to_features,
)
from ner_crf.model.arg_parse import get_args

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

MODEL_CONFIG_CLASSES = list(MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys())
                  for conf in MODEL_CONFIG_CLASSES), ())

TOKENIZER_ARGS = ["do_lower_case", "strip_accents", "keep_accents", "use_fast"]


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)




def evaluate(
    args,
    model,
    tokenizer,
    labels,
    pad_token_label_id,
    mode,
    prefix="",
):
    eval_dataset = load_and_cache_examples(args,
                                           tokenizer,
                                           labels,
                                           pad_token_label_id,
                                           mode=mode)

    args.eval_batch_size = args.per_gpu_eval_batch_size
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset,
                                 sampler=eval_sampler,
                                 batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3]
            }
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet"] else None
                )  # XLM and RoBERTa don"t use segment_ids
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids,
                                      inputs["labels"].detach().cpu().numpy(),
                                      axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=2)

    label_map = {i: label for i, label in enumerate(labels)}

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id:
                out_label_list[i].append(label_map[out_label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])

    results = {
        "loss": eval_loss,
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
    }

    logger.info("***** Eval results %s *****", prefix)
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))

    return results, preds_list


def load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode):
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}".format(
            mode,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length)),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s",
                    cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        examples = read_examples_from_file(args.data_dir, mode)
        features = convert_examples_to_features(
            examples,
            labels,
            args.max_seq_length,
            tokenizer,
            cls_token_at_end=bool(args.model_type in ["xlnet"]),
            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=bool(args.model_type in ["roberta"]),
            # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=bool(args.model_type in ["xlnet"]),
            # pad on the left for xlnet
            pad_token=tokenizer.pad_token_id,
            pad_token_segment_id=tokenizer.pad_token_type_id,
            pad_token_label_id=pad_token_label_id,
        )

        logger.info("Saving features into cached file %s",
                    cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features],
                                 dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features],
                                  dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features],
                                   dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features],
                                 dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                            all_label_ids)
    return dataset


def setup_general_config(args):
    """Setup general config before train or test.

    Config including: \n
        1. Check for output directory
        2. Setup CUDA and GPU
        3. Setup logging
        4. Setup seed
    """

    # Check for output directory
    if (os.path.exists(args.output_dir) and os.listdir(args.output_dir)
            and args.do_train and not args.overwrite_output_dir):
        raise ValueError(
            f"Output directory ({args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome.")

    # Setup CUDA and GPU
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.warning("Process device: %s, 16-bits training: %s", device,
                   args.fp16)

    # Set seed
    set_seed(args)


def evaluate_pipeline(args, labels, pad_token_label_id, tokenizer_args):
    results = {}
    tokenizer = AutoTokenizer.from_pretrained(args.output_dir,
                                              **tokenizer_args)
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
        global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
        model = AutoModelForTokenClassification.from_pretrained(checkpoint)
        model.to(args.device)
        result, _ = evaluate(args,
                             model,
                             tokenizer,
                             labels,
                             pad_token_label_id,
                             mode="dev",
                             prefix=global_step)
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
    return results


def predict_pipeline(args, labels, pad_token_label_id, tokenizer_args):
    tokenizer = AutoTokenizer.from_pretrained(args.output_dir,
                                              **tokenizer_args)
    model = AutoModelForTokenClassification.from_pretrained(args.output_dir)
    model.to(args.device)
    result, predictions = evaluate(args,
                                   model,
                                   tokenizer,
                                   labels,
                                   pad_token_label_id,
                                   mode="test")
    # Save results
    output_test_results_file = os.path.join(args.output_dir,
                                            "test_results.txt")
    with open(output_test_results_file, "w") as writer:
        for key in sorted(result.keys()):
            writer.write("{} = {}\n".format(key, str(result[key])))
    # Save predictions
    output_test_predictions_file = os.path.join(args.output_dir,
                                                "test_predictions.txt")
    with open(output_test_predictions_file, "w") as writer:
        with open(os.path.join(args.data_dir, "test.txt"), "r") as f_obj:
            example_id = 0
            for line in f_obj:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    writer.write(line)
                    if not predictions[example_id]:
                        example_id += 1
                elif predictions[example_id]:
                    output_line = line.split(
                    )[0] + " " + predictions[example_id].pop(0) + "\n"
                    writer.write(output_line)
                else:
                    logger.warning(
                        "Maximum sequence length exceeded: No prediction for '%s'.",
                        line.split()[0])


def train_pipeline(args, labels, pad_token_label_id, tokenizer_args):
    num_labels = len(labels)
    id2label = {str(i): label for i, label in enumerate(labels)}
    label2id = {label: i for i, label in enumerate(labels)}
    cache_dir = args.cache_dir if args.cache_dir else None
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
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


def main():
    # Part 1
    args = get_args(MODEL_TYPES, ALL_MODELS)
    setup_general_config(args)

    # Prepare for DuEE task
    labels = get_labels(args.labels)

    # Use cross entropy ignore index as padding label id ,
    # so that only real label ids contribute to the loss later
    pad_token_label_id = CrossEntropyLoss().ignore_index

    # Load pretrained model and tokenizer
    args.model_type = args.model_type.lower()

    tokenizer_args = {
        k: v
        for k, v in vars(args).items() \
        if v is not None and k in TOKENIZER_ARGS
    }
    logger.info("Tokenizer arguments: %s", tokenizer_args)

    logger.info("Training/evaluation parameters %s", args)

    # NOTE: Training
    if args.do_train:
        train_pipeline(args, labels, pad_token_label_id, tokenizer_args)

    # NOTE: Evaluation
    results = {}
    if args.do_eval:
        results = evaluate_pipeline(args, labels, pad_token_label_id,
                                    tokenizer_args)

    # NOTE: Inference
    if args.do_predict:
        predict_pipeline(args, labels, pad_token_label_id, tokenizer_args)

    return results


if __name__ == "__main__":
    main()
