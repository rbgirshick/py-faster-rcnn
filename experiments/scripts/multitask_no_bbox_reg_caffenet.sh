#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/multitask_no_bbox_reg_caffenet.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/test_net.py --gpu $1 \
  --def models/CaffeNet/test.prototxt \
  --net output/default/voc_2007_trainval/caffenet_fast_rcnn_iter_40000.caffemodel \
  --imdb voc_2007_test \
  --cfg experiments/cfgs/no_bbox_reg.yml
