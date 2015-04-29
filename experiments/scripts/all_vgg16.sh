#!/bin/bash

set -e
GPU=$1
NET=vgg16
./experiments/scripts/default_${NET}.sh $GPU
./experiments/scripts/fc_only_${NET}.sh $GPU
./experiments/scripts/multitask_no_bbox_reg_${NET}.sh $GPU
./experiments/scripts/no_bbox_reg_${NET}.sh $GPU
./experiments/scripts/piecewise_${NET}.sh $GPU
./experiments/scripts/svd_${NET}.sh $GPU
./experiments/scripts/svm_${NET}.sh $GPU
