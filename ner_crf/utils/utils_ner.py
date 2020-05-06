# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Named entity recognition fine-tuning: utilities to work with CoNLL-2003 task. """

import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Union

import torch
from torch import nn
from torch.utils.data.dataset import Dataset

from transformers import PreTrainedTokenizer, torch_distributed_zero_first
from transformers.data.data_collator import DefaultDataCollator
from transformers.tokenization_bert import _is_whitespace

logger = logging.getLogger(__name__)


@dataclass
class SpanInputExample():
    event_id: str
    text: str
    roles: List[Dict]
    trigger: Dict
    sent_ori_2_new: Dict[int]


@dataclass
class SpanInputFeatures:
    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: List[int]
    start_labels_seq: List[List[int]] = None
    end_labels_seq: List[List[int]] = None
    tok_to_new: List[List[int]]
    new_to_tok: List[int]


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


class SpanDataset(Dataset):
    features: List[SpanInputFeatures]
    pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index

    def __init__(
            self,
            data_dir: str,
            tokenizer: PreTrainedTokenizer,
            label2id: Dict[str, int],
            max_seq_length: Optional[int] = None,
            overwrite_cache=False,
            mode: Split = Split.train,
            local_rank=-1,
    ):
        # Load data features from cache or dataset file
        basename = "cached_{}_{}_{}".format(mode.value,
                                            tokenizer.__class__.__name__,
                                            str(max_seq_length))
        cached_features_file = os.path.join(data_dir, basename)

        with torch_distributed_zero_first(local_rank):
            # Make sure only the first process in distributed training processes the dataset,
            # and the others will use the cache.

            if os.path.exists(cached_features_file) and not overwrite_cache:
                logger.info("Loading features from cached file %s",
                            cached_features_file)
                self.features = torch.load(cached_features_file)
            else:
                logger.info("Creating features from dataset file at %s",
                            data_dir)
                examples = span_read_examples_from_file(data_dir, mode)
                logger.info("Number of examples: %s", len(examples))
                # TODO clean up all this to leverage built-in features of tokenizers
                self.features = span_convert_examples_to_features(
                    examples, label2id, max_seq_length, tokenizer, mode)
                if local_rank in [-1, 0]:
                    logger.info("Saving features into cached file %s",
                                cached_features_file)
                    torch.save(self.features, cached_features_file)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> SpanInputFeatures:
        return self.features[i]


def str_full_to_width(ustring):
    """把字符串全角转半角"""
    ss = []
    for s in ustring:
        rstring = ""
        for uchar in s:
            inside_code = ord(uchar)
            if inside_code == 12288:  # 全角空格直接转换
                inside_code = 32
            elif (inside_code >= 65281
                  and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
                inside_code -= 65248
            rstring += chr(inside_code)
        ss.append(rstring)
    return ''.join(ss)


def process_sent_ori_2_new(text, roles_list, trigger):
    """process_sent_ori_2_new"""
    words = list(text)
    sent_ori_2_new_index = {}
    new_words = []
    new_roles_list = {}
    for role_type, role in roles_list.items():
        new_roles_list[role_type] = {
            "role_type": role_type,
            "start": -1,
            "end": -1
        }
    new_trigger = {
        "trigger": trigger['trigger'],
        'start': trigger['start'],
        'end': trigger['end']
    }

    for i, w in enumerate(words):
        for role_type, role in roles_list.items():
            if i == role["start"]:
                new_roles_list[role_type]["start"] = len(new_words)
            if i == role["end"]:
                new_roles_list[role_type]["end"] = len(new_words)
        # Trigger
        if i == trigger['start']:
            new_trigger['start'] = len(new_words)
        if i == trigger['end']:
            new_trigger['end'] = len(new_words)

        if not w.strip():
            sent_ori_2_new_index[i] = -1
            for role_type, role in roles_list.items():
                if i == role["start"]:
                    new_roles_list[role_type]["start"] += 1
                if i == role["end"]:
                    new_roles_list[role_type]["end"] -= 1
            # Trigger
            if i == trigger['start']:
                new_trigger['start'] += 1
            if i == trigger['end']:
                new_trigger['end'] -= 1
        else:
            sent_ori_2_new_index[i] = len(new_words)
            new_words.append(w)

    for role_type, role in new_roles_list.items():
        if role["start"] > -1:
            role["text"] = u"".join(new_words[role["start"]:role["end"] + 1])
        if role["end"] == len(new_words):
            role["end"] = len(new_words) - 1

    if new_trigger['start'] > -1:
        new_trigger['start'] = u"".join(
            new_words[new_trigger["start"]:new_trigger["end"] + 1])
    if new_trigger['end'] == len(new_words):
        new_trigger['end'] = len(new_words) - 1

    return new_words, sent_ori_2_new_index, new_roles_list, new_trigger


def span_read_examples_from_file(data_dir, mode: Union[Split, str]
                                 ) -> List[SpanInputExample]:
    if isinstance(mode, Split):
        mode = mode.value
    file_path = os.path.join(data_dir, f"{mode}.json")

    examples = []
    with open(file_path, encoding="utf-8") as f_obj:
        for _, line in enumerate(f_obj):
            line = str_full_to_width(line)
            json_obj = json.loads(line.strip())
            event_id = json_obj["event_id"]
            text = json_obj["text"]

            # Role
            roles_list = {}
            for role in json_obj["arguments"]:
                role_type = role["role"]
                role_text = role["argument"]
                role_start = role["argument_start_index"]
                role_end = role_start + len(role_text) - 1
                roles_list[role_type] = {
                    "role_type": role_type,
                    "start": role_start,
                    "end": role_end,
                    "argument": role_text
                }

            # Trigger
            trigger = {
                "trigger":
                json_obj['trigger'],
                "start":
                json_obj["trigger_start_index"],
                "end":
                json_obj["trigger_start_index"] + len(json_obj['trigger']) - 1
            }
            (
                new_words,
                sent_ori_2_new,
                new_role_list,
                new_trigger,
            ) = process_sent_ori_2_new(text, roles_list, trigger)

            examples.append(
                SpanInputExample(
                    event_id=event_id,
                    text=u"".join(new_words),
                    roles=new_role_list,
                    trigger=new_trigger,
                    sent_ori_2_new=sent_ori_2_new,
                ))

    return examples


def _stem(token):
    if token[:2] == '##':
        return token[2:]
    else:
        return token


def _is_special(token: str):
    return bool(token) \
            and token.startswith('[') \
            and token.endswith(']')


def _get_mapping_from_new_and_tok(text, tokens):
    new_to_tok, tok_to_new, offset = [], [], 0
    for i, token in enumerate(tokens):
        if _is_special(token):
            tok_to_new.append([offset, offset])
            new_to_tok.append(i)
            offset += 1
        else:
            token = _stem(token)
            start = text[offset:].index(token) + offset
            end = start + len(token)
            tok_to_new.append([start, end - 1])
            new_to_tok.extend([i] * len(token))
            offset = end
    return new_to_tok, tok_to_new


def span_convert_example_to_features(
        example: List[SpanInputExample],
        label2id: Dict[str, int],
        max_seq_length: int,
        tokenizer: PreTrainedTokenizer,
        mode: Union[Split, str],
) -> SpanInputFeatures:
    assert not any(_is_whitespace(c) for c in example.text)

    text = example.text
    tokens = tokenizer.tokenize(text)
    if tokenizer.do_lower_case:
        text = text.lower()
    new_to_tok, tok_to_new = _get_mapping_from_new_and_tok(text, tokens)

    assert len(tok_to_new) == len(tokens)
    assert len(new_to_tok) == len(text)

    start_positions, end_positions = None, None
    if mode == Split.train:
        start_positions = [[] for _ in tokens]
        end_positions = [[] for _ in tokens]

        for _, role in example.roles.items():
            tok_start = new_to_tok[role['start']]
            tok_end = new_to_tok[role['end']]
            start_positions[tok_start].append(label2id[role['role_type']])
            end_positions[tok_end].append(label2id[role['role_type']])

    # TODO: handle if trigger is out of boundary
    encode_dict = tokenizer.encode_plus(
        text=tokens,
        max_length=max_seq_length,
        pad_to_max_length=True,
        return_token_type_id=True,
        return_attention_mask=True,
    )

    # As paper said:
    # "Therefore, we feed argument extractor with the segment ids of trigger tokens being one."
    tok_trigger_start = new_to_tok[example.trigger['start']]
    tok_trigger_end = new_to_tok[example.trigger['end']]
    for idx in range(tok_trigger_start, tok_trigger_end + 1):
        # FIXME: use other method like attention to handle trigger out of boundary.
        if idx >= max_seq_length:
            break
        encode_dict['token_type_ids'][idx] = 1

    # Construct labels for each positions
    def _construct_labels_seq(positions):
        positions.insert(0, [])
        pad_len = max_seq_length - len(positions)
        positions.extend([[] for _ in range(pad_len)])

        # Convert to (seq_len, num_label)
        labels_seq = []
        for position in positions:
            labels = [0] * len(label2id)
            for idx in position:
                labels[idx] = 1
            labels_seq.append(labels)
        return labels_seq

    start_labels_seq = end_labels_seq = None
    if start_positions:
        if tokenizer.pad_token_id not in encode_dict['input_ids']:
            start_positions = start_positions[:tokenizer.
                                              max_len_single_sentence]
            end_positions = end_positions[:tokenizer.max_len_single_sentence]
        start_labels_seq = _construct_labels_seq(start_positions)
        end_labels_seq = _construct_labels_seq(end_positions)

    encode_dict['start_labels_seq'] = start_labels_seq
    encode_dict['end_labels_seq'] = end_labels_seq
    encode_dict['tok_to_new'] = tok_to_new
    encode_dict['new_to_tok'] = new_to_tok

    res = {
        'feature': SpanInputFeatures(**encode_dict),
        'all_doc_tokens': tokens,
        'start_positions': start_positions,
        'end_positions': end_positions
    }

    return res


def span_convert_examples_to_features(
        examples: List[SpanInputExample],
        label2id: Dict[str, int],
        max_seq_length: int,
        tokenizer: PreTrainedTokenizer,
        mode: Union[Split, str],
) -> List[SpanInputFeatures]:
    """ Loads a data file into a list of `SpanInputFeatures`
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    # TODO clean up all this to leverage built-in features of tokenizers

    features = []
    for ex_index, example in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))
        res = span_convert_example_to_features(example, label2id,
                                               max_seq_length, tokenizer, mode)
        feature = res['feature']
        features.append(feature)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("event_id: %s", example.event_id)
            logger.info("trigger: %s", example.trigger)
            logger.info("roles: %s", example.roles)
            logger.info("tokens: %s", " ".join(res['all_doc_tokens']))
            logger.info("input_ids: %s", str(feature.input_ids))
            logger.info("input_mask: %s", str(feature.attention_mask))
            logger.info("segment_ids: %s", str(feature.token_type_ids))
            logger.info("start_positions: %s", str(res['start_positions']))
            logger.info("end_positions: %s", str(res['end_positions']))

    return features


def get_labels(path: str) -> List[str]:
    if path:
        with open(path, "r") as f_obj:
            labels = f_obj.read().splitlines()
        return labels
    else:
        return [
            "O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG",
            "B-LOC", "I-LOC"
        ]


@dataclass
class SpanDataController(DefaultDataCollator):
    """
    Very simple data collator that:
    - simply collates batches of dict-like objects
    - Performs special handling for potential keys named:
        - `label`: handles a single value (int or float) per object
        - `label_ids`: handles a list of values per object
    - does not do any additional preprocessing

    i.e., Property names of the input object will be used as corresponding inputs to the model.
    See glue and ner for example of how it's useful.
    """
    def collate_batch(self, features: List[SpanInputFeatures]
                      ) -> Dict[str, torch.Tensor]:
        # pylint: disable=not-callable
        # In this method we'll make the assumption that all `features` in the batch
        # have the same attributes.
        # So we will look at the first element as a proxy for what attributes exist
        # on the whole batch.
        first = features[0]

        # Special handling for labels.
        # Ensure that tensor is created with the correct type
        # (it should be automatically the case, but let's make sure of it.)
        if all(hasattr(first, "start_labels_seq"), hasattr(first, "end_labels_seq")) \
                and all(first.start_labels_seq, first.end_labels_seq):
            start_labels_seq = torch.tensor(
                [f.start_labels_seq for f in features], dtype=torch.float)
            end_labels_seq = torch.tensor([f.end_labels_seq for f in features],
                                          dtype=torch.float)
            batch = {"labels": torch.stack(start_labels_seq, end_labels_seq)}
        else:
            batch = {}

        # Handling of all other possible attributes.
        # Again, we will use the first element to figure out which key/values are not None for this model.
        for key, val in vars(first).items():
            if key not in ("start_labels_seq", "end_labels_seq", "tok_to_orig_indices") \
                    and val is not None and not isinstance(val, str):
                batch[key] = torch.tensor([getattr(f, key) for f in features],
                                          dtype=torch.long)
        return batch
