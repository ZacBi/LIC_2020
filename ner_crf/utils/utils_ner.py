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
from transformers.tokenization_bert import whitespace_tokenize, _is_whitespace
from transformers.data.data_collator import DefaultDataCollator

logger = logging.getLogger(__name__)


@dataclass
class SpanInputExample():
    event_id: str
    text: str
    words: List[str]
    char_to_word_indices: List[int]
    roles: List[Dict]
    trigger: Dict


@dataclass
class SpanInputFeatures:
    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: List[int]
    tok_to_orig_indices: List[int]
    start_labels_seq: tuple[List[List[int]]] = None
    end_labels_seq: tuple[List[List[int]]] = None


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
                logger.info(
                    f"Loading features from cached file {cached_features_file}"
                )
                self.features = torch.load(cached_features_file)
            else:
                logger.info(
                    f"Creating features from dataset file at {data_dir}")
                examples = span_read_examples_from_file(data_dir, mode)
                # TODO clean up all this to leverage built-in features of tokenizers
                self.features = span_convert_examples_to_features(
                    examples, label2id, max_seq_length, tokenizer, mode)
                if local_rank in [-1, 0]:
                    logger.info(
                        f"Saving features into cached file {cached_features_file}"
                    )
                    torch.save(self.features, cached_features_file)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> SpanInputFeatures:
        return self.features[i]


def span_read_examples_from_file(data_dir, mode: Union[Split, str]
                                 ) -> List[SpanInputExample]:
    if isinstance(mode, Split):
        mode = mode.value
    file_path = os.path.join(data_dir, f"{mode}.json")

    examples = []
    with open(file_path, encoding="utf-8") as f_obj:
        for idx, line in enumerate(f_obj):
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
                "trigger": json_obj['trigger'],
                "start": json_obj["trigger_start_index"],
                "end": json_obj["trigger_start_index"] + len(trigger) - 1
            }

            words = []
            char_to_word_indices = []
            prev_is_whitespace = True
            # Seperate word by whitespace
            for i, char in enumerate(text):
                # Remove the influence of whitespace
                if _is_whitespace(char):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        words.append(char)
                        prev_is_whitespace = False
                    else:
                        words[-1].append(char)
                char_to_word_indices.append(len(words) - 1)

                examples.append(
                    SpanInputExample(
                        event_id=event_id,
                        text=text,
                        words=words,
                        char_to_word_indices=char_to_word_indices,
                        roles=roles_list,
                        trigger=trigger,
                    ))

    return examples


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
        if ex_index % 10_000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))
        res = span_convert_example_to_features(example, label2id,
                                               max_seq_length, tokenizer, mode)
        if res is None:
            continue
        feature = res['feature']
        features.append(feature)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("event_id: %s", example.event_id)
            logger.info("tokens: %s", " ".join(res['all_doc_tokens']))
            logger.info("input_ids: %s", " ".join(feature.input_ids))
            logger.info("input_mask: %s", " ".join(feature.attention_mask))
            logger.info("segment_ids: %s", " ".join(feature.token_type_ids))
            logger.info("start_positions: %s",
                        " ".join(res['start_positions']))
            logger.info("end_positions: %s", " ".join(res['end_positions']))

    return features


def span_convert_example_to_features(
        example: List[SpanInputExample],
        label2id: Dict[str, int],
        max_seq_length: int,
        tokenizer: PreTrainedTokenizer,
        mode: Union[Split, str],
) -> SpanInputFeatures:

    # Check if trigger or role in text
    if mode == Split.train:
        actual_text = " ".join(example.words)
        trigger = " ".join(whitespace_tokenize(example.trigger['trigger']))
        if actual_text.find(trigger) == -1:
            logger.warning("Invalid data, id: %s, text: %s, argument: %s",
                           example.event_id, actual_text, trigger)
            return None
        for role in example.roles:
            argument = " ".join(whitespace_tokenize(role['argument']))
            if actual_text.find(argument) == -1:
                logger.warning("Invalid data, id: %s, text: %s, argument: %s",
                               example.event_id, actual_text, argument)
            return None

    tok_to_orig_indices = []
    orig_to_tok_indices = []
    all_doc_tokens = []
    for i, token in enumerate(example.words):
        orig_to_tok_indices.append(len(all_doc_tokens))
        for sub_token in tokenizer.tokenize(token):
            tok_to_orig_indices.append(i)
            all_doc_tokens.append(sub_token)

    # Get token start and token end according to original start and end
    def _get_token_start_and_end(start, end):
        word_start, word_end = c2w_indices[start], c2w_indices[end]
        tok_start = orig_to_tok_indices[word_start]
        if word_end < len(example.words) - 1:
            tok_end = orig_to_tok_indices[word_end + 1] - 1
        else:
            tok_end = len(all_doc_tokens) - 1

    c2w_indices = example.char_2_word_indices
    start_positions, end_positions = None, None
    if mode == Split.train:
        start_positions = [[] for _ in all_doc_tokens]
        end_positions = [[] for _ in all_doc_tokens]

        for role in example.roles:
            tok_start, tok_end = _get_token_start_and_end(
                role['start'], role['end'])
            start_positions[tok_start].append(label2id[role['role_type']])
            end_positions[tok_end].append(label2id[role['role_type']])

    # TODO: handle if trigger is out of boundary
    encode_dict = tokenizer.encode_plus(
        text=all_doc_tokens,
        max_length=max_seq_length,
        pad_to_max_length=True,
        return_token_type_id=True,
        return_attention_mask=True,
    )

    # As paper said:
    # "Therefore, we feed argument extractor with the segment ids of trigger tokens being one."
    tok_trigger_start, tok_trigger_end = _get_token_start_and_end(
        example.trigger['start'], example.trigger['end'])
    for idx in range(tok_trigger_start, tok_trigger_end + 1):
        # FIXME: use other method like attention to handle trigger out of boundary.
        if idx > max_seq_length:
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
        pad_idx = encode_dict['input_ids'].find(tokenizer.pad_token_id)
        if pad_idx == -1:
            start_positions = start_positions[:tokenizer.
                                              max_len_single_sentence]
            end_positions = end_positions[:tokenizer.max_len_single_sentence]
        start_labels_seq = _construct_labels_seq(start_positions)
        end_labels_seq = _construct_labels_seq(end_positions)

    encode_dict['start_labels_seq'] = start_labels_seq
    encode_dict['end_labels_seq'] = end_labels_seq
    encode_dict['tok_to_orig_indices'] = tok_to_orig_indices

    res = {
        'feature': SpanInputFeatures(**encode_dict),
        'all_doc_tokens': all_doc_tokens,
        'start_positions': start_positions,
        'end_positions': end_positions
    }

    return res


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
    def collate_batch(self, features: List[InputDataClass]
                      ) -> Dict[str, torch.Tensor]:
        # In this method we'll make the assumption that all `features` in the batch
        # have the same attributes.
        # So we will look at the first element as a proxy for what attributes exist
        # on the whole batch.
        first = features[0]

        # Special handling for labels.
        # Ensure that tensor is created with the correct type
        # (it should be automatically the case, but let's make sure of it.)
        if all(hasattr(first,"start_labels_seq"), hasattr(first,"end_labels_seq")) \
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
        for k, v in vars(first).items():
            if k not in ("start_labels_seq", "end_labels_seq", "tok_to_orig_indices") \
                    and v is not None and not isinstance(v, str):
                batch[k] = torch.tensor([getattr(f, k) for f in features],
                                        dtype=torch.long)
        return batch
