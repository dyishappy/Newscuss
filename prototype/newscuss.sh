#!/bin/bash

model="kogpt"
cache_dir="home/dai/cache"
client="fjiawyhfk;j.afjhaj.dfjksahflh/2jhdkflhsljkzfj"

echo "Starting using ${model} for summarize"

CUDA_VISIBLE_DEVICES=0 python src/newscuss.py \
    --model $model \
    --cache_dir $cache_dir \
    --client $client ;

# available models: gpt, clova, kogpt, blc, kobart
# ☠︎ clova and kogpt are not finished yet.