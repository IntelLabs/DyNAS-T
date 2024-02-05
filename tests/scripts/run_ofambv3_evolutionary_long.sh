#!/usr/bin/env bash

source $( dirname -- "$0"; )/config.sh

time ${RUN_COMMAND} \
        --results_path ${RESULTS_PATH}/results_ofambv3_evolutionary_long.csv \
        --supernet ofa_mbv3_d234_e346_k357_w1.0 \
        --dataset_path  ${DATASET_IMAGENET_PATH} \
        --search_tactic evolutionary \
        --population ${LONG_LINAS_POPULATION} \
        --batch_size ${BATCH_SIZE} \
        --seed ${SEED} \
        --measurements macs accuracy_top1 \
        --num_evals ${LONG_LINAS_NUM_EVALS} \
        --device ${DEVICE} \
        --test_fraction ${TEST_FRACTION}
