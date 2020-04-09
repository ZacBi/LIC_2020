# coding=utf-8
# pylint: disable=bad-continuation

import os
import logging

from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
)

from ner_crf.model.evaluate import evaluate

logger = logging.getLogger(__file__)  # pylint: disable=invalid-name


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
