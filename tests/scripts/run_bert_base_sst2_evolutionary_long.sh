#!/usr/bin/env bash

source $( dirname -- "$0"; )/config.sh

time ${RUN_COMMAND} \
        --results_path ${RESULTS_PATH}/results_bert_base_sst2_evolutionary_long.csv \
        --supernet bert_base_sst2 \
        --supernet_ckpt_path ${CHECKPOINT_BERT_BASE_SST2_PATH} \
        --dataset_path  ${DATASET_SST2_PATH} \
        --search_tactic evolutionary \
        --population ${LONG_LINAS_POPULATION} \
        --batch_size ${BATCH_SIZE} \
        --seed ${SEED} \
        --measurements macs accuracy_sst2 latency params \
        --optimization_metrics macs accuracy_sst2 \
        --num_evals ${LONG_LINAS_NUM_EVALS} \
        --device ${DEVICE} \
        --test_fraction ${TEST_FRACTION}
