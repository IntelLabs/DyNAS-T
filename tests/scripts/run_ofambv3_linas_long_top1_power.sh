#!/usr/bin/env bash

source $( dirname -- "$0"; )/config.sh

time mpirun \
    --report-bindings \
    -x MASTER_ADDR=127.0.0.1 \
    -x MASTER_PORT=1234 \
    -np 1 \
    -bind-to socket \
    -map-by socket \
    python dynast/cli.py \
        --results_path ${RESULTS_PATH}/results_ofa_mbv3_power_nodram_linas_long.csv \
        --optimization_metrics accuracy_top1 energy \
        --measurements accuracy_top1 macs params latency energy \
        --seed ${SEED} \
        --batch_size ${BATCH_SIZE} \
        --search_tactic linas \
        --population ${LONG_LINAS_POPULATION} \
        --num_evals ${LONG_LINAS_NUM_EVALS}
