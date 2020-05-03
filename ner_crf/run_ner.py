import logging
import os
from typing import Dict

import numpy as np

from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from ner_crf.models import BertCRF, BertSpan
from ner_crf.processors import Trainer, EvalPrediction
from ner_crf.utils import (
    DataTrainingArguments,
    ModelArguments,
    SpanDataset,
    SpanDataController,
    Split,
    get_labels,
)

MODEL_CLASSES = {'crf': BertCRF, 'span': BertSpan}

logger = logging.getLogger(__name__)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir) and training_args.do_train
            and not training_args.overwrite_output_dir):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            "Use --overwrite_output_dir to overcome.")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
        if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Prepare for task
    labels = get_labels(data_args.labels)
    id2label = {i: label for i, label in enumerate(labels)}
    label2id = {label: i for i, label in enumerate(labels)}
    num_labels = len(labels)

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast,
    )
    model_class = MODEL_CLASSES[model_args.model_class]
    model = model_class.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # Get datasets
    train_dataset = (SpanDataset(
        data_dir=data_args.data_dir,
        tokenizer=tokenizer,
        label2id=label2id,
        max_seq_length=data_args.max_seq_length,
        overwrite_cache=data_args.overwrite_cache,
        mode=Split.train,
        local_rank=training_args.local_rank,
    ) if training_args.do_train else None)

    eval_dataset = (SpanDataset(
        data_dir=data_args.data_dir,
        tokenizer=tokenizer,
        label2id=label2id,
        max_seq_length=data_args.max_seq_length,
        overwrite_cache=data_args.overwrite_cache,
        mode=Split.dev,
        local_rank=training_args.local_rank,
    ) if training_args.do_eval else None)

    def calculate_precision_recall_f1(num_label, num_infer, num_correct):
        """calculate_f1"""
        if num_infer == 0:
            precision = 0.0
        else:
            precision = num_correct * 1.0 / num_infer

        if num_label == 0:
            recall = 0.0
        else:
            recall = num_correct * 1.0 / num_label

        if num_correct == 0:
            f1_score = 0.0
        else:
            f1_score = 2 * precision * recall / (precision + recall)
        return precision, recall, f1_score

    def compute_metrics(p: EvalPrediction) -> Dict:  # pylint: disable=invalid-name
        """Here we compute `wordpiece level` precision, recall and f1"""
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        pred_start_probs = sigmoid(p.predictions)[:, 0]
        pred_end_probs = sigmoid(p.predictions)[:, 1]
        pred_role_span_lists = model.arg_span_determine(
            pred_start_probs, pred_end_probs, p.last_valid_indices)
        true_role_span_lists = model.arg_span_determine(
            p.label_ids[:, 0], p.label_ids[:, 1], p.last_valid_indices)

        common = pred = annotate = 0
        for pred_span, true_span in zip(pred_role_span_lists,
                                        true_role_span_lists):
            _len = lambda span: span[1] - span[0] + 1 if span else 0
            pred += _len(pred_span)
            annotate += _len(true_span)
            if all(
                    pred_span,
                    true_span,
                    max(pred_span[0], true_span[0]) <= min(
                        pred_span[1], true_span[1]),
            ):
                common += min(pred_span[1], true_span[1]) - max(
                    pred_span[0], true_span[0]) + 1

        precision, recall, f1_score = calculate_precision_recall_f1(
            annotate, pred, annotate)
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1_score,
        }

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=SpanDataController,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        trainer.train(model_path=model_args.model_name_or_path if os.path.
                      isdir(model_args.model_name_or_path) else None)
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval and training_args.local_rank in [-1, 0]:
        logger.info("*** Evaluate ***")

        result = trainer.evaluate()

        output_eval_file = os.path.join(training_args.output_dir,
                                        "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key, value in result.items():
                logger.info("  %s = %s", key, value)
                writer.write("%s = %s\n" % (key, value))

            results.update(result)

    # Predict
    # if training_args.do_predict and training_args.local_rank in [-1, 0]:
    #     test_dataset = SpanDataset(
    #         data_dir=data_args.data_dir,
    #         tokenizer=tokenizer,
    #         label2id=label2id,
    #         max_seq_length=data_args.max_seq_length,
    #         overwrite_cache=data_args.overwrite_cache,
    #         mode=Split.test,
    #         local_rank=training_args.local_rank,
    #     )

    #     predictions, label_ids, metrics = trainer.predict(test_dataset)
    #     preds_list, _ = align_predictions(predictions, label_ids)

    #     output_test_results_file = os.path.join(training_args.output_dir,
    #                                             "test_results.txt")
    #     with open(output_test_results_file, "w") as writer:
    #         for key, value in metrics.items():
    #             logger.info("  %s = %s", key, value)
    #             writer.write("%s = %s\n" % (key, value))

    #     # Save predictions
    #     output_test_predictions_file = os.path.join(training_args.output_dir,
    #                                                 "test_predictions.txt")
    #     with open(output_test_predictions_file, "w") as writer:
    #         with open(os.path.join(data_args.data_dir, "test.txt"), "r") as f:
    #             example_id = 0
    #             for line in f:
    #                 if line.startswith(
    #                         "-DOCSTART-") or line == "" or line == "\n":
    #                     writer.write(line)
    #                     if not preds_list[example_id]:
    #                         example_id += 1
    #                 elif preds_list[example_id]:
    #                     output_line = line.split(
    #                     )[0] + " " + preds_list[example_id].pop(0) + "\n"
    #                     writer.write(output_line)
    #                 else:
    #                     logger.warning(
    #                         "Maximum sequence length exceeded: No prediction for '%s'.",
    #                         line.split()[0])

    return results


if __name__ == "__main__":
    main()
