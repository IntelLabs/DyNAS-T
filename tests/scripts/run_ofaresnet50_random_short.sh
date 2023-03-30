#!/usr/bin/env bash

source $( dirname -- "$0"; )/config.sh

time python run_search.py \
        --results_path ${RESULTS_PATH}/results_ofaresnet50_random_short.csv \
        --supernet ofa_resnet50 \
        --dataset_path  ${DATASET_IMAGENET_PATH} \
        --search_tactic random \
        --population ${SHORT_RANDOM_POPULATION} \
        --valid_size ${SHORT_RANDOM_VALID_SIZE} \
        --seed ${SEED} \
        --measurements macs accuracy_top1 \
        --num_evals ${SHORT_RANDOM_POPULATION}
