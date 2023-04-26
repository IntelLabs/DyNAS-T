#!/usr/bin/env bash

for i in {1..10};
do
    python check_accuracy_resnet50.py 2>/dev/null | grep lat;
done;
