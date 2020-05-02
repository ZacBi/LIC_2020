#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
处理样本样本数据作为训练以及处理schema
"""
import os
import sys
import json
import random
import logging
from collections import defaultdict
from ner_crf.utils import utils_lic

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def schema_event_type_process():
    """schema_process"""
    schema_path = sys.argv[2]
    save_path = sys.argv[3]
    if not schema_path or not save_path:
        raise Exception("set schema_path and save_path first")
    index = 0
    event_types = set()
    for line in utils_lic.read_by_lines(schema_path):
        d_json = json.loads(line)
        event_types.add(d_json["event_type"])

    outputs = []
    for event_type in list(event_types):
        outputs.append(u"B-{}\t{}".format(event_type, index))
        index += 1
        outputs.append(u"I-{}\t{}".format(event_type, index))
        index += 1
    outputs.append(u"O\t{}".format(index))
    print(u"include event type {},  create label {}".format(
        len(event_types), len(outputs)))
    utils_lic.write_by_lines(save_path, outputs)


def schema_role_process():
    """schema_role_process"""
    schema_path = sys.argv[2]
    save_path = sys.argv[3]
    if not schema_path or not save_path:
        raise Exception("set schema_path and save_path first")
    index = 0
    roles = set()
    for line in utils_lic.read_by_lines(schema_path):
        d_json = json.loads(line)
        for role in d_json["role_list"]:
            roles.add(role["role"])
    # Set labels
    outputs = []
    outputs.append(u"O\t{}".format(0))
    index += 1
    for role in list(roles):
        outputs.append(u"B-{}\t{}".format(role, index))
        index += 1
        outputs.append(u"I-{}\t{}".format(role, index))
        index += 1
    # Add "[CLS]" and "[SEP]" for CRF

    outputs.append(u"[CLS]\t{}".format(index))
    outputs.append(u"[SEP]\t{}".format(index + 1))
    print(u"include roles {}，create label {}".format(len(roles), len(outputs)))
    utils_lic.write_by_lines(save_path, outputs)


def origin_events_process():
    """origin_events_process

    Args:
        alias: the prefix of data, e.g. 'alias_train.json'

    """
    origin_events_path = sys.argv[2]
    save_dir = sys.argv[3]
    try:
        split_data = sys.argv[4]
        split_data = split_data.lower == 'true'
    except:  # pylint: disable=bare-except
        split_data = False
    if not origin_events_path or not save_dir:
        raise Exception("set origin_events_path and save_dir first")
    output = []
    lines = utils_lic.read_by_lines(origin_events_path)

    # Process lines
    for line in lines:
        d_json = json.loads(line)
        for event in d_json["event_list"]:
            event["event_id"] = u"{}_{}".format(d_json["id"], event["trigger"])
            event["text"] = d_json["text"]
            event["id"] = d_json["id"]
            output.append(json.dumps(event, ensure_ascii=False))

    if split_data:
        random.shuffle(output)  # 随机一下
        # 按照 8 / 2 分
        train_data_len = int(len(output) * 0.8)
        train_data = output[:train_data_len]
        test_data = output[train_data_len:]
        print(
            u"include sentences {}, events {}, train data {}, dev data {}, test data {}"
            .format(len(lines), len(output), len(train_data), len(test_data),
                    len(test_data)))
        utils_lic.write_by_lines(u"{}/train.json".format(save_dir), train_data)
        utils_lic.write_by_lines(u"{}/dev.json".format(save_dir), test_data)
        utils_lic.write_by_lines(u"{}/test.json".format(save_dir), test_data)
    else:
        utils_lic.write_by_lines(u"{}/train.json".format(save_dir), output)


def statistic():
    """Observe the data distution in dataset.

    Statistic:
        1. 每一短句事件数量分布
        2. 每一类型(event_type)数量分布
        3. argument overlap(同一事件类型论元多角色, 不同事件类型论元多角色)
        4. 论元角色数量匹配schema程度
    """
    fpath = '/mnt/d/Workspace/Github/mine/LIC_2020/data/ori_datasets/train.json'
    schema_path = '/mnt/d/Workspace/Github/mine/LIC_2020/data/ori_datasets/event_schema.json'

    # Construct schema
    schema = {}
    lines = utils_lic.read_by_lines(schema_path)
    for line in lines:
        j_obj = json.loads(line)
        if line in j_obj:
            continue
        schema[j_obj['event_type']] = j_obj

    # dist: distribution
    empty_argument_ids = []
    sent_count = event_count = inner_overlap = 0
    cross_overlap = defaultdict(int)
    event_list_len_dist = defaultdict(int)
    event_type_num_dist = defaultdict(int)
    match_dist = {'match': 0, 'not_match': 0}
    loose_match_dist = {'match': 0, 'not_match': 0}
    lines = utils_lic.read_by_lines(fpath)
    for line in lines:
        sent_count += 1
        j_obj = json.loads(line)
        # Distribution of the num of event per sentence
        event_list_len_dist[len(j_obj['event_list'])] += 1
        for event in j_obj['event_list']:
            event_count += 1
            event_type = event['event_type']
            # Distribution of num of event type
            event_type_num_dist[event_type] += 1
            num_schema_event_role = len(schema[event_type]['role_list'])
            num_sent_event_role = len(event['arguments'])

            # Strict match
            if num_schema_event_role == num_sent_event_role:
                match_dist['match'] += 1
            else:
                match_dist['not_match'] += 1

            # Loose match don't count for `time` field
            num_arg_no_time = num_sent_event_role
            if any(arg['role'] == '时间' for arg in event['arguments']):
                num_arg_no_time -= 1
            if num_schema_event_role - 1 == num_arg_no_time:
                loose_match_dist['match'] += 1
            else:
                loose_match_dist['not_match'] += 1

        # NOTE: Check for argument overlap
        s1_count, s2_count = check_argument_overlap(
            j_obj['id'],
            len(j_obj['text']),
            j_obj['event_list'],
            empty_argument_ids,
        )
        inner_overlap += s1_count
        for key, val in s2_count.items():
            cross_overlap[key] += val

    # Count the num of each event class
    event_class_num_dist = defaultdict(int)
    for event_type, val in event_type_num_dist.items():
        event_class = event_type.split('-')[0]
        event_class_num_dist[event_class] += val

    res = {
        'sent_count': sent_count,
        'event_count': event_count,
        'event_list_len': dict(event_list_len_dist),
        'event_type_num': dict(event_type_num_dist),
        'event_class_num': dict(event_class_num_dist),
        'strict_role_match': dict(match_dist),
        'loose_role_match': dict(loose_match_dist),
        'inner_arg_overlap': inner_overlap,
        'cross_arg_overlap': dict(cross_overlap),
        'empty_argument_ids': list(set(empty_argument_ids)),
    }
    dirname = os.path.dirname(fpath)
    with open(os.path.join(dirname, 'train_dist.json'), 'w') as f_obj:
        f_obj.write(json.dumps(res, ensure_ascii=False))


# TODO: Find overlap by span.
def check_argument_overlap(text_id, len_sent, event_list, empty_argument_ids):
    """Check for argument overlap

    Type of argument overlap:
        1. one argument play more than one role in a sentence.
        2. one argument play more than one role in different sentences.

    Returns:
        s1_count (int): count for situation 1.
        s2_count (defaultdict): count for situation 2.

    NOTE: Take argument 'time' for consideration and just think about index.
    """
    states = []
    s1_count = max_idx = 0
    s2_count = defaultdict(int)

    if not event_list:
        logger.warning("%s has empty event list.", text_id)

    # Check foe situation 1
    for event in event_list:
        state = 0
        idx_group = list(a['argument_start_index'] for a in event['arguments'])

        if not idx_group:
            logger.warning("%s has empty arguments.", text_id)
            empty_argument_ids.append(text_id)
            continue

        max_idx = max(max(idx_group), max_idx)
        if len(idx_group) != len(set(idx_group)):
            s1_count += 1
        for idx in idx_group:
            state ^= 1 << idx
        states.append(state)

    # Check for situation 2
    for idx in range(max_idx):
        sum_ = sum((state >> idx) & 1 for state in states)
        if sum_ > 1:
            s2_count[sum_] += 1
    return s1_count, s2_count


def run(func_name=None):
    """main"""
    func_mapping = {
        "origin_events_process": origin_events_process,
        "schema_event_type_process": schema_event_type_process,
        "schema_role_process": schema_role_process,
        "statistic": statistic
    }
    func_name = sys.argv[1]
    if func_name not in func_mapping:
        raise Exception("no function {}, please select [ {} ]".format(
            func_name, u" | ".join(func_mapping.keys())))
    func_mapping[func_name]()


if __name__ == '__main__':
    statistic()
