import os
import logging
import random
import json
from pathlib import Path

import torch
import numpy as np

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def seed_everything(seed=1024):
    '''
    设置整个开发环境的seed
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def save_model(args, model, tokenizer, optimizer, scheduler, global_step):
    # Save model checkpoint
    output_dir = os.path.join(args.output_dir, "ckpt-{}".format(global_step))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Take care of distributed/parallel training
    model_to_save = (model.module if hasattr(model, "module") else model)
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    torch.save(args, os.path.join(output_dir, "training_args.bin"))
    logger.info("Saving model checkpoint to %s", output_dir)
    torch.save(optimizer.state_dict(), os.path.join(output_dir,
                                                    "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(output_dir,
                                                    "scheduler.pt"))
    logger.info("Saving optimizer and scheduler states to %s", output_dir)


def json_to_text(file_path, data):
    '''
    将json list写入text文件中
    :param file_path:
    :param data:
    :return:
    '''
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    with open(str(file_path), 'w') as f_obj:
        for line in data:
            line = json.dumps(line, ensure_ascii=False)
            f_obj.write(line + '\n')
