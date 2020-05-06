#!/bin/bash

python ner_crf/run_ner.py \
    --model_class span \
    --model_name_or_path data/models/chinese-bert-wwm-ext/ \
    --cache_dir data/cache \
    --data_dir data/datasets \
    --labels data/datasets/vocab_roles_label_map_span.txt \
    --max_seq_length 256 \
    --output_dir data/outputs/0504 \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --evaluate_during_training \
    --per_gpu_train_batch_size 32 \
    --eval_batch_size 32 \
    --learning_rate 3e-4 \
    --num_train_epochs 10 \
    --warmup_steps 400 \
    --logging_steps 50 \
    --save_steps 500 \
    --save_total_limit 5 \
