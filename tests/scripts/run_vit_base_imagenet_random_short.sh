#!/usr/bin/env bash

source $( dirname -- "$0"; )/config.sh

time ${RUN_COMMAND} \
        --results_path ${RESULTS_PATH}/results_vit_base_imagenet_random_short.csv \
        --supernet vit_base_imagenet \
        --supernet_ckpt_path ${CHECKPOINT_VIT_BASE_IMAGENET_PATH} \
        --dataset_path  ${DATASET_IMAGENET_PATH} \
        --search_tactic random \
        --population ${SHORT_RANDOM_POPULATION} \
        --test_fraction ${SHORT_RANDOM_IMAGENET_TEST_FRACTION} \
        --seed ${SEED} \
        --measurements macs accuracy_top1 \
        --num_evals ${SHORT_RANDOM_POPULATION}
