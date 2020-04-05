import json

import pandas as pd
from sklearn.preprocessing import LabelEncoder

def feed(args,tokenizer):
    train_x,train_y = data_process(args.train_file,tokenizer)
    dev_x,dev_y = data_process(args.dev_file,tokenizer)

    tag_to_ix = {} 
    for tag in train_y + dev_y:
        if tag not in tag_to_ix:
            tag_to_ix[tag] = len(tag_to_ix) + 1

    train_y = [tag_to_ix[i] for i in train_y]
    dev_y = [tag_to_ix[i] for i in dev_y]

    train_data = []
    for i,j in zip(train_x,train_y):
        train_data.append([i,j])

    dev_data = []
    for i,j in zip(dev_x,dev_y):
        dev_data.append([i,j])

    if args.do_debug:
        train_data = train_data[0:100]
        dev_data = dev_data[0:100]

    return train_data, dev_data, len(tag_to_ix) + 1


def data_process(path,tokenizer):
    f = open(path, encoding='utf-8')
    data = []
    for line in f.readlines():
        dic = json.loads(line)
        data.append(dic)

    event_list = []
    text = []
    id = []
    for item in data:
        event_list.append(item["event_list"])
        text.append(item["text"])
        id.append(item["id"])
    
    arguments = []
    class_type = []
    event_type = []
    trigger = []
    trigger_start_index = []
    for item in event_list:
        arguments.append(item[0]["arguments"])
        class_type.append(item[0]["class"])
        event_type.append(item[0]["event_type"])
        trigger.append(item[0]["trigger"])
        trigger_start_index.append(item[0]["trigger_start_index"])

    
    tokenized_text = []
    for t in text:
        tokenized_text.append(tokenizer.tokenize(t))
    indexed_tokens = []
    for text in tokenized_text:
        indexed_tokens.append(tokenizer.convert_tokens_to_ids(text))
    
    return indexed_tokens,class_type
