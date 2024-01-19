#!/usr/bin/env bash

source $( dirname -- "$0"; )/config.sh

time ${RUN_COMMAND} \
        --results_path ${RESULTS_PATH}/results_beit_imagenet_random_short.csv \
        --supernet beit3_imagenet \
        --supernet_ckpt_path ${CHECKPOINT_BEIT_IMAGENET_PATH} \
        --dataset_path  ${DATASET_IMAGENET_PATH} \
        --search_tactic random \
        --batch_size ${BATCH_SIZE} \
        --population ${SHORT_RANDOM_POPULATION} \
        --test_fraction ${SHORT_RANDOM_IMAGENET_TEST_FRACTION} \
        --seed ${SEED} \
        --measurements macs accuracy_top1 \
        --num_evals ${SHORT_RANDOM_POPULATION} \
        --device ${DEVICE}
