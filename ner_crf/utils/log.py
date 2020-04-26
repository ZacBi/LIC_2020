import logging
from typing import Dict

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def logging_train(args: Dict, num_examples: int, total_steps: int):
    """Logging in train"""

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", num_examples)
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


def logging_continuing_training(args, epochs_trained, global_step,
                                steps_trained_in_current_epoch):
    """Check if continuing training from a checkpoint"""

    logger.info(
        "Continuing training from checkpoint, will skip to saved global_step")
    logger.info("Continuing training from epoch %d", epochs_trained)
    logger.info("Continuing training from global step %d", global_step)
    logger.info("Will skip the first %d steps in the first epoch",
                steps_trained_in_current_epoch)


def logging_evaluation(results):
    pass
