# coding=utf-8
# pylint: disable=bad-continuation
import glob
import os
import logging

import torch
from torch.utils.data import (
    DataLoader,
    SequentialSampler,
    TensorDataset,
)

from sklearn.metrics import (
    f1_score,
    recall_score,
    precision_score,
)

from transformers import (
    WEIGHTS_NAME,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

import numpy as np
from tqdm import tqdm

from ner_crf.utils.utils_ner import (
    read_examples_from_file,
    convert_examples_to_features,
)

logger = logging.getLogger(__file__)  # pylint: disable=invalid-name


def load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode):
    # pylint: disable=not-callable
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
        # Original examples, it's a list including InputExample instance
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
            # roberta uses an extra separator b/w pairs of sentences,
            # cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
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


def evaluate(args,
             model,
             tokenizer,
             labels,
             pad_token_label_id,
             mode,
             prefix=""):
    eval_dataset = load_and_cache_examples(
        args,
        tokenizer,
        labels,
        pad_token_label_id,
        mode=mode,
    )

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
        "precision": precision_score(out_label_list, preds_list, average='micro'),
        "recall": recall_score(out_label_list, preds_list, average='micro'),
        "f1": f1_score(out_label_list, preds_list, average='micro'),
    }

    logger.info("***** Eval results %s *****", prefix)
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))

    return results, preds_list


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
