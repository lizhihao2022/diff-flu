#!/bin/bash
python main.py \
    --model FNO \
    --layers 4 \
    --dim 128 \
    --cuda True \
    --device 4 \
    --epochs 300 \
    --patience 10 \
    --eval_freq 5 \
    --batch_size 32 \
    --optimizer Adam \
    --lr 0.0001 \
    --log True\
    --weight_decay 0.002 \
    --momentum 0.95 \
    --dataset km_flow \
    --data_dir ./data/km_flow/ \
    --train_ratio 0.6 \
    --valid_ratio 0.2 \
    --test_ratio 0.2 \
