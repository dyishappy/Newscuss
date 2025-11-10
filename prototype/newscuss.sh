#!/bin/bash

# available models: gpt, clova, kogpt, blc, kobart, solar, deepseek
model="gpt"
cache_dir="/nas/datahub/HFHOME/hub"
client="???"

echo "Starting using ${model} for summarize"

CUDA_VISIBLE_DEVICES=3 python src/newscuss.py \
    --model $model \
    --cache_dir $cache_dir \
    --client $client ;

# solar is unfinished