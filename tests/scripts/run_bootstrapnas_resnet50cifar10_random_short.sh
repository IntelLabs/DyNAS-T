#!/usr/bin/env bash

source $( dirname -- "$0"; )/config.sh

time ${RUN_COMMAND} \
        --results_path ${RESULTS_PATH}/results_bootstrapnas_resnet50cifar10_random_short.csv \
        --supernet bootstrapnas_resnet50_cifar10 \
        --dataset_path  ${DATASET_CIFAR10_PATH} \
        --search_tactic random \
        --population ${SHORT_RANDOM_POPULATION} \
        --valid_size ${SHORT_RANDOM_VALID_SIZE} \
        --seed ${SEED} \
        --measurements macs accuracy_top1 \
        --num_evals ${SHORT_RANDOM_POPULATION}
