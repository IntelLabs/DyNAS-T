#!/usr/bin/env bash

source $( dirname -- "$0"; )/config.sh

time ${RUN_COMMAND} \
        --results_path ${RESULTS_PATH}/results_beit_imagenet_linas_long.csv \
        --supernet beit3_imagenet \
        --supernet_ckpt_path ${CHECKPOINT_BEIT_IMAGENET_PATH} \
        --dataset_path  ${DATASET_IMAGENET_PATH} \
        --search_tactic linas \
        --batch_size ${BATCH_SIZE} \
        --population ${LONG_LINAS_POPULATION} \
        --seed ${SEED} \
        --measurements latency macs params accuracy_top1 \
        --optimization_metrics latency accuracy_top1 \
        --num_evals ${LONG_LINAS_NUM_EVALS} \
        --device ${DEVICE}
