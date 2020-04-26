""" Named entity recognition fine-tuning: utilities to work with CLUENER task. """

import os
import json
import logging
from collections import namedtuple

import torch
from ner_crf.processors.DataProcessor import DataProcessor
from ner_crf.utils import whitespace_tokenize

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

InputFeatures = namedtuple(
    'InputFeatures',
    ['input_ids', 'input_mask', 'token_type_ids', 'label_ids'])

Example = namedtuple('Example', [
    "id", "text_a", "label", "ori_text", "ori_2_new_index", "roles", "sentence"
])


def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    all_input_ids, all_attention_mask, all_token_type_ids, all_labels = map(
        torch.stack, zip(*batch))
    all_lengths = all_attention_mask.sum(1)
    max_len = torch.max(all_lengths)
    all_input_ids = all_input_ids[:, :max_len]
    all_attention_mask = all_attention_mask[:, :max_len]
    all_token_type_ids = all_token_type_ids[:, :max_len]
    all_labels = all_labels[:, :max_len]
    return all_input_ids, all_attention_mask, all_token_type_ids, all_labels


def _reseg_token_label(tokens, labels, tokenizer):
    """_reseg_token_label"""
    assert len(tokens) == len(labels)
    ret_tokens = []
    ret_labels = []
    for token, label in zip(tokens, labels):
        sub_token = tokenizer.tokenize(token)
        if not sub_token:
            continue
        ret_tokens.extend(sub_token)
        if len(sub_token) == 1:
            ret_labels.append(label)
            continue

        if label == "O" or label.startswith("I-"):
            ret_labels.extend([label] * len(sub_token))
        elif label.startswith("B-"):
            i_label = "I-" + label[2:]
            ret_labels.extend([label] + [i_label] * (len(sub_token) - 1))

    assert len(ret_tokens) == len(ret_labels)
    return ret_tokens, ret_labels


# FIXME: Just for BERT, create a base for different models
def convert_examples_to_features(examples,
                                 tokenizer,
                                 label2id,
                                 max_seq_length=128):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        # Resegment tokens and labels
        tokens = whitespace_tokenize(example.text_a)
        labels = whitespace_tokenize(example.label)
        tokens, labels = _reseg_token_label(tokens, labels, tokenizer)

        # Account for [CLS] and [SEP] with "- 2".
        special_tokens_count = 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[:(max_seq_length - special_tokens_count)]
            labels = labels[:(max_seq_length - special_tokens_count)]

        tokens = [tokenizer.cls_token] + tokens
        labels = [tokenizer.cls_token] + labels
        tokens += [tokenizer.sep_token]
        labels += [tokenizer.sep_token]

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        label_ids = [label2id[label] for label in labels]
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * padding_length
        input_mask += [0] * padding_length
        token_type_ids += [tokenizer.pad_token_id] * padding_length
        label_ids += [tokenizer.pad_token_id] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(token_type_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        feature = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=token_type_ids,
            label_ids=label_ids,
        )
        logging_examples(feature, ex_index)
        features.append(feature)
    return features


class CluenerProcessor(DataProcessor):
    """Processor for the chinese ner data set."""
    def __init__(self, label_vocab=None):
        self.label_vocab = label_vocab

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "test.json")), "test")

    # FIXME: dynamic generate labels
    def get_labels(self, label_vocab):
        """See base class."""
        return []

    def _read_json(self, input_file):
        """_read_json_file"""
        input_data = []
        with open(input_file, "r", encoding='utf8') as f_obj:
            for line in f_obj:
                d_json = json.loads(line.strip())
                input_data.append(d_json)
        return input_data

    def _create_examples(self, input_data, set_type):
        """_examples_by_json"""
        def process_sent_ori_2_new(sent, roles_list):
            """process_sent_ori_2_new"""
            words = list(sent)
            sent_ori_2_new_index = {}
            new_words = []
            new_roles_list = {}
            for role_type, role in roles_list.items():
                new_roles_list[role_type] = {
                    "role_type": role_type,
                    "start": -1,
                    "end": -1
                }

            for i, word in enumerate(words):
                for role_type, role in roles_list.items():
                    if i == role["start"]:
                        new_roles_list[role_type]["start"] = len(new_words)
                    if i == role["end"]:
                        new_roles_list[role_type]["end"] = len(new_words)

                if not word.strip():
                    # delete the white space
                    sent_ori_2_new_index[i] = -1
                    for role_type, role in roles_list.items():
                        if i == role["start"]:
                            new_roles_list[role_type]["start"] += 1
                        if i == role["end"]:
                            new_roles_list[role_type]["end"] -= 1
                else:
                    sent_ori_2_new_index[i] = len(new_words)
                    new_words.append(word)

            for role_type, role in new_roles_list.items():
                if role["start"] > -1:
                    role["text"] = u"".join(
                        new_words[role["start"]:role["end"] + 1])
                if role["end"] == len(new_words):
                    role["end"] = len(new_words) - 1

            return [words, new_words, sent_ori_2_new_index, new_roles_list]

        examples = []
        for idx, data in enumerate(input_data):
            event_id = data["event_id"]
            sentence = data["text"]
            roles_list = {}
            for role in data["arguments"]:
                role_type = role["role"]
                role_start = role["argument_start_index"]
                role_text = role["argument"]
                role_end = role_start + len(role_text) - 1
                roles_list[role_type] = {
                    "role_type": role_type,
                    "start": role_start,
                    "end": role_end,
                    "argument": role_text
                }

            (sent_words, new_sent_words, ori_2_new_sent_index,
             new_roles_list) = process_sent_ori_2_new(sentence.lower(),
                                                      roles_list)

            new_sent_labels = [u"O"] * len(new_sent_words)
            for role_type, role in new_roles_list.items():
                for i in range(role["start"], role["end"] + 1):
                    if i == role["start"]:
                        new_sent_labels[i] = u"B-{}".format(role_type)
                    else:
                        new_sent_labels[i] = u"I-{}".format(role_type)
            example = Example(id=event_id,
                              text_a=u" ".join(new_sent_words),
                              label=u" ".join(new_sent_labels),
                              ori_text=sent_words,
                              ori_2_new_index=ori_2_new_sent_index,
                              roles=new_roles_list,
                              sentence=sentence)

            logging_examples(example, idx)

            examples.append(example)
        return examples


def logging_examples(example, idx, idx_bound=5):
    if idx >= idx_bound:
        return
    logger.info("******** example %d ********", idx)
    for key, val in example._asdict().items():
        logger.info('%s : %s', key, val)
