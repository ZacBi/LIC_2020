""" Named entity recognition fine-tuning: utilities to work with CLUENER task. """
# pylint: disable=bad-continuation

import os
import json
import logging
from collections import namedtuple

import torch
from ner_crf.processors.utils_ner import DataProcessor
from ner_crf.tokenizer import whitespace_tokenize

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

InputFeatures = namedtuple(
    'InputFeatures',
    ['input_ids', 'input_mask', 'segment_ids', 'label_ids', 'input_len'])

Example = namedtuple('Example', [
    "id", "text_a", "label", "ori_text", "ori_2_new_index", "roles", "sentence"
])


def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels = map(
        torch.stack, zip(*batch))
    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_attention_mask = all_attention_mask[:, :max_len]
    all_token_type_ids = all_token_type_ids[:, :max_len]
    all_labels = all_labels[:, :max_len]
    return all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_lens


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


def convert_examples_to_features(
    examples,
    label_map,
    max_seq_length,
    tokenizer,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
):
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

        tokens += [sep_token]
        labels += [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            labels += [cls_token]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            label_ids = [cls_token] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        label_ids = [label_map[label] for label in labels]
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        input_len = len(label_ids)
        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] *
                          padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] *
                           padding_length) + segment_ids
            label_ids = ([pad_token] * padding_length) + label_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s",
                        " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s",
                        " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          input_len=input_len,
                          segment_ids=segment_ids,
                          label_ids=label_ids))
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
    for key, val in example._asdict():
        logger.info('%s : %s', key, val)
