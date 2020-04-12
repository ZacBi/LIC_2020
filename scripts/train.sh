#!/bin/bash
export ROOT="~/work/LIC_2020"


python ner_crf/run_ner_crf.py \
--data_dir data/datasets \
--model_type bert \
--model_name_or_path data/models/bert-base-chinese \
--output_dir data/outputs/0412 \
--label_map_config data/datasets/vocab_roles_label_map.txt \
--cache_dir data/cache \
--do_train \
--evaluate_during_training \
