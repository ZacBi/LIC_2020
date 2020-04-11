#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=bad-continuation
"""
处理样本样本数据作为训练以及处理schema
"""
import json
import random

from ner_crf.utils import utils_lic


def schema_event_type_process(schema_path=None, save_path=None):
    """schema_process"""
    if not schema_path or not save_path:
        raise Exception("set schema_path and save_path first")
    index = 0
    event_types = set()
    for line in utils_lic.read_by_lines(schema_path):
        d_json = json.loads(line)
        event_types.add(d_json["event_type"])

    outputs = []
    for et in list(event_types):
        outputs.append(u"B-{}\t{}".format(et, index))
        index += 1
        outputs.append(u"I-{}\t{}".format(et, index))
        index += 1
    outputs.append(u"O\t{}".format(index))
    print(u"include event type {},  create label {}".format(
        len(event_types), len(outputs)))
    utils_lic.write_by_lines(save_path, outputs)


def schema_role_process(schema_path=None, save_path=None):
    """schema_role_process"""
    if not schema_path or not save_path:
        raise Exception("set schema_path and save_path first")
    index = 0
    roles = set()
    for line in utils_lic.read_by_lines(schema_path):
        d_json = json.loads(line)
        for role in d_json["role_list"]:
            roles.add(role["role"])
    outputs = []
    for r in list(roles):
        outputs.append(u"B-{}\t{}".format(r, index))
        index += 1
        outputs.append(u"I-{}\t{}".format(r, index))
        index += 1
    outputs.append(u"O\t{}".format(index))
    print(u"include roles {}，create label {}".format(len(roles), len(outputs)))
    utils_lic.write_by_lines(save_path, outputs)


def origin_events_process(
    origin_events_path=None,    
    save_dir=None,
    split_data=False,
    alias='',
):
    """origin_events_process

    Args:
        alias: the prefix of data, e.g. 'alias_train.json'

    """
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
            u"include sentences {}, events {}, train datas {}, dev datas {}, test datas {}"
            .format(len(lines), len(output), len(train_data), len(test_data),
                    len(test_data)))
        utils_lic.write_by_lines(u"{}/train_{}.json".format(save_dir, alias),
                             train_data)
        utils_lic.write_by_lines(u"{}/dev_{}.json".format(save_dir, alias),
                             test_data)
        utils_lic.write_by_lines(u"{}/test_{}.json".format(save_dir, alias),
                             test_data)
    else:
        utils_lic.write_by_lines(u"{}/train_{}.json".format(save_dir, alias),
                             output)


def run(func_name=None):
    """main"""
    func_mapping = {
        "origin_events_process": origin_events_process,
        "schema_event_type_process": schema_event_type_process,
        "schema_role_process": schema_role_process
    }
    if func_name not in func_mapping:
        raise Exception("no function {}, please select [ {} ]".format(
            func_name, u" | ".join(func_mapping.keys())))
    func_mapping[func_name]()


if __name__ == '__main__':
    run()
