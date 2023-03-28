#!/usr/bin/env bash

source $( dirname -- "$0"; )/config.sh

time python run_search.py \
        --results_path ${RESULTS_PATH}/results_ofaresnet50_random_long.csv \
        --supernet ofa_resnet50 \
        --dataset_path  ${DATASET_IMAGENET_PATH} \
        --search_tactic linas \
        --population ${LONG_LINAS_POPULATION} \
        --seed ${SEED} \
        --measurements macs accuracy_top1 \
        --num_evals ${LONG_LINAS_NUM_EVALS}
