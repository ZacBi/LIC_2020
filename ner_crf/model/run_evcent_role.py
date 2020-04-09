# coding=utf-8
# pylint: disable=bad-continuation

import os
import random
import logging

# torch group
import torch
from torch.nn import CrossEntropyLoss

# transformers
from transformers import MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING

import numpy as np

# Self package
from ner_crf.utils.utils_ner import get_labels
from ner_crf.model.arg_parse import get_args
from ner_crf.model.evaluate import evaluate_pipeline
from ner_crf.model.train import train_pipeline
from ner_crf.model.infer import predict_pipeline

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
