#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
帮助类
"""
import hashlib


def read_by_lines(path, encoding="utf-8"):
    """read the data by line"""
    result = list()
    with open(path, "r", encoding=encoding) as f_obj:
        for line in f_obj:
            result.append(line.strip())
    return result


def write_by_lines(path, data, t_code="utf-8"):
    """write the data"""
    with open(path, "w", encoding=t_code) as outfile:
        outfile.writelines(d + '\n' for d in data)


def cal_md5(string):
    """calculate string md5"""
    string = string.encode("utf-8", "ignore")
    return hashlib.md5(string).hexdigest()
