# pylint: disable=bad-continuation
import os
import logging
from typing import Dict

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def logging_train(args: Dict, num_examples: int, total_steps: int):
    """Logging in train"""

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(num_examples))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size * args.gradient_accumulation_steps,
    )
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", total_steps)


def logging_continuing_training(args, num_batchs: int):
    """Check if continuing training from a checkpoint"""
    if os.path.exists(args.model_name_or_path) \
                and "ckpt" in args.model_name_or_path:
        # set global_step to global_step of last saved checkpoint from model path
        global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (num_batchs //
                                         args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (
            num_batchs // args.gradient_accumulation_steps)

        logger.info(
            "Continuing training from checkpoint, will skip to saved global_step"
        )
        logger.info("Continuing training from epoch %d", epochs_trained)
        logger.info("Continuing training from global step %d", global_step)
        logger.info("Will skip the first %d steps in the first epoch",
                    steps_trained_in_current_epoch)


def logging_evaluation(results):
    pass
