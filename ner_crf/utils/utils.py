#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
帮助类
"""
import hashlib
import unicodedata


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


def _stem(token):
    """获取token的“词干”（如果是##开头，则自动去掉##）
    """
    if token[:2] == '##':
        return token[2:]
    else:
        return token


def _is_special(char):
    """判断是不是有特殊含义的符号
    """
    return bool(char) and (char[0] == '[') and (char[-1] == ']')


def _is_control(char):
    """控制类字符判断
    """
    return unicodedata.category(char) in ('Cc', 'Cf')


def rematch(text, tokens, do_lower_case=True):
    """给出原始的text和tokenize后的tokens的映射关系
    """

    normalized_text, char_mapping = '', []
    for i, char in enumerate(text):
        if do_lower_case:
            char = unicodedata.normalize('NFD', char)
            char = ''.join(
                [c for c in char if unicodedata.category(c) != 'Mn'])
            char = char.lower()
        char = ''.join([
            c for c in char
            if not (ord(c) == 0 or ord(c) == 0xfffd or _is_control(c))
        ])
        normalized_text += char
        char_mapping.extend([i] * len(char))

    text, token_mapping, offset = normalized_text, [], 0
    for token in tokens:
        if _is_special(token):
            token_mapping.append([])
        else:
            token = _stem(token)
            start = text[offset:].index(token) + offset
            end = start + len(token)
            token_mapping.append(char_mapping[start:end])
            offset = end

    return token_mapping


if __name__ == "__main__":
    text = '魅族新机，将于 5 月 30 日发布，命名为魅族 16Xs，官方宣传海报写有 “165g OF BALANCE” 的海报图，似乎暗示该机重量为 165g。'
    tokens = [
        '魅', '族', '新', '机', '，', '将', '于', '5', '月', '30', '日', '发', '布', '，',
        '命', '名', '为', '魅', '族', '16', '##x', '##s', '，', '官', '方', '宣', '传',
        '海', '报', '写', '有', '[UNK]', '165', '##g', 'of', 'balance', '[UNK]',
        '的', '海', '报', '图', '，', '似', '乎', '暗', '示', '该', '机', '重', '量', '为',
        '165', '##g', '。'
    ]
    mapping = rematch(text, tokens)
