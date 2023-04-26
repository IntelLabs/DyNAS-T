#!/usr/bin/env bash

for i in {1..10};
do
    OMP_NUM_THREADS=14 python check_accuracy_resnet50.py 2>/dev/null | grep lat;
done;
