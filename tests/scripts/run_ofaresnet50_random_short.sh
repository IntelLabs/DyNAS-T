#!/usr/bin/env bash

source $( dirname -- "$0"; )/config.sh

time ${RUN_COMMAND} \
        --results_path ${RESULTS_PATH}/results_ofaresnet50_random_short.csv \
        --supernet ofa_resnet50 \
        --dataset_path  ${DATASET_IMAGENET_PATH} \
        --search_tactic random \
        --population ${SHORT_RANDOM_POPULATION} \
        --test_size ${SHORT_RANDOM_TEST_SIZE} \
        --batch_size ${BATCH_SIZE} \
        --seed ${SEED} \
        --measurements macs accuracy_top1 \
        --num_evals ${SHORT_RANDOM_POPULATION}
