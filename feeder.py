import json

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def feed(args,tokenizer):
    f = open(args.train_file, encoding='utf-8')
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
    x = []
    for text in tokenized_text:
        x.append(tokenizer.convert_tokens_to_ids(text))
    
    tag_to_ix = {} 
    for tag in class_type:
        if tag not in tag_to_ix:
            tag_to_ix[tag] = len(tag_to_ix) + 1
    y = [tag_to_ix[i] for i in class_type]

    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)
    train_data = []
    for i,j in zip(X_train,y_train):
        train_data.append([i,j])
    dev_data = []
    for i,j in zip(X_test,y_test):
        dev_data.append([i,j])

    if args.do_debug:
        train_data = train_data[0:100]
        dev_data = train_data[0:20]

    return train_data, dev_data , len(tag_to_ix) + 1
