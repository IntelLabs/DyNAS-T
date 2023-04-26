#!/usr/bin/env bash

for i in {1..10};
do
    mpirun \
        --report-bindings \
        -x MASTER_ADDR=127.0.0.1 \
        -x MASTER_PORT=1238 \
        -np 1 \
        -bind-to socket \
        -map-by socket \
            python check_accuracy_resnet50.py 2>/dev/null | grep lat;
done;
