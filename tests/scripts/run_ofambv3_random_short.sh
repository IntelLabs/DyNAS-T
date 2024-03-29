#!/usr/bin/env bash

source $( dirname -- "$0"; )/config.sh

time ${RUN_COMMAND} \
        --results_path ${RESULTS_PATH}/results_ofambv3_random_short.csv \
        --supernet ofa_mbv3_d234_e346_k357_w1.0 \
        --dataset_path  ${DATASET_IMAGENET_PATH} \
        --search_tactic random \
        --population ${SHORT_RANDOM_POPULATION} \
        --batch_size ${BATCH_SIZE} \
        --test_fraction ${SHORT_RANDOM_IMAGENET_TEST_FRACTION} \
        --seed ${SEED} \
        --measurements macs accuracy_top1 \
        --num_evals ${SHORT_RANDOM_POPULATION} \
        --device ${DEVICE}
