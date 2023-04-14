#!/usr/bin/env bash

RUN_COMMAND="python dynast/cli.py"
SEED=37
RESULTS_PATH="/tmp"
DATASET_IMAGENET_PATH="/datasets/imagenet-ilsvrc2012/"


########################################################################################################
# SHORT runs config. Shoud use random search tactic to allow for a very limited number of evaluations. #
########################################################################################################

# The number of evaluations is set by SHORT_RANDOM_POPULATION and should be an even number to allow for a distributed test.
SHORT_RANDOM_POPULATION=2

# Limit the number of validation samples to speed up the test.
SHORT_RANDOM_TEST_SIZE=20


#################################################################################################
# LONG runs config. Shoud use LINAS search tactic and use parameters as the end-user would use. #
#################################################################################################
LONG_LINAS_POPULATION=50
LONG_LINAS_NUM_EVALS=250
