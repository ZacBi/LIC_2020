#!/bin/bash

python ner_crf/run_ner_crf.py \
    --data_dir data/datasets \
    --model_type bert \
    --model_name_or_path data/models/chinese-bert-wwm-ext/ \
<<<<<<< HEAD
    --output_dir data/outputs/0425 \
=======
    --output_dir data/outputs/0419 \
>>>>>>> 3acfc8135191463552bc668d7de93e63b653f92e
    --label_map_config data/datasets/vocab_roles_label_map.txt \
    --cache_dir data/cache \
    --do_train \
    --evaluate_during_training \
    --per_gpu_train_batch_size 80 \
    --per_gpu_eval_batch_size 64 \
    --eval_max_seq_length 128 \
    --learning_rate 3e-4 \
    --crf_learning_rate 0.2 \
    --num_train_epochs 15 \
    --warmup_ratio 0.1 \
    --save_steps 500 \
    --overwrite_output_dir
