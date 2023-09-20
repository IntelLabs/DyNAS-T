#!/usr/bin/env bash

RUN_COMMAND="python dynast/cli.py"
SEED=37
RESULTS_PATH="/tmp"
DATASET_IMAGENET_PATH="/datasets/imagenet-ilsvrc2012/"
DATASET_CIFAR10_PATH="/tmp/cifar10/"
DEVICE="cpu"
BATCH_SIZE=128
TEST_FRACTION=1.0

CHECKPOINT_VIT_BASE_IMAGENET_PATH="/tmp/vit/checkpoint.pth.tar"


########################################################################################################
# SHORT runs config. Shoud use random search tactic to allow for a very limited number of evaluations. #
########################################################################################################

# The number of evaluations is set by SHORT_RANDOM_POPULATION and should be an even number to allow for a distributed test.
SHORT_RANDOM_POPULATION=2

# Limit the number of validation samples to speed up the test.
SHORT_RANDOM_IMAGENET_TEST_FRACTION=0.005


#################################################################################################
# LONG runs config. Shoud use LINAS search tactic and use parameters as the end-user would use. #
#################################################################################################
LONG_LINAS_POPULATION=50
LONG_LINAS_NUM_EVALS=250
