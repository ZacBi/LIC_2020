#!/bin/bash

python ner_crf/run_ner_crf.py \
--data_dir data/datasets \
--model_type bert \
--model_name_or_path data/models/bert-base-chinese \
--output_dir data/outputs/0412 \
--label_map_config data/datasets/vocab_roles_label_map.txt \
--cache_dir data/cache \
--do_train \
--evaluate_during_training \
--per_gpu_train_batch_size 128 \
--per_gpu_eval_batch_size 64 \
--eval_max_seq_length 128 \
--learning_rate 3e-5 \
--num_train_epochs 15 \
--save_steps 200 \
--overwrite_output_dir 