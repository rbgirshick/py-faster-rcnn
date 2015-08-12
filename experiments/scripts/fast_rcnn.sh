#!/bin/bash
# Usage:
# ./experiments/scripts/default.sh GPU NET [options args to {train,test}_net.py]
# Example:
# ./experiments/scripts/default.sh 0 CaffeNet \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=$2
NET_lc=${NET,,}

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:2:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

LOG="experiments/logs/default_${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu ${GPU_ID} \
  --solver models/${NET}/fast_rcnn/solver.prototxt \
  --weights data/imagenet_models/${NET}.v2.caffemodel \
  --imdb voc_2007_trainval \
  ${EXTRA_ARGS}

set +x
NET_FINAL=`grep -B 1 "done solving" ${LOG} | grep "Wrote snapshot" | awk '{print $4}'`
set -x

time ./tools/test_net.py --gpu ${GPU_ID} \
  --def models/${NET}/fast_rcnn/test.prototxt \
  --net ${NET_FINAL} \
  --imdb voc_2007_test \
  ${EXTRA_ARGS}
