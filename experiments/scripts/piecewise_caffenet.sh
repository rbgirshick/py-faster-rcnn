#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/piecewise_caffenet.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu $1 \
  --solver models/CaffeNet/piecewise/solver.prototxt \
  --weights output/no_bbox_reg/voc_2007_trainval/caffenet_fast_rcnn_no_bbox_reg_iter_40000.caffemodel \
  --imdb voc_2007_trainval \
  --cfg experiments/cfgs/piecewise.yml

time ./tools/test_net.py --gpu $1 \
  --def models/CaffeNet/test.prototxt \
  --net output/piecewise/voc_2007_trainval/caffenet_fast_rcnn_piecewise_iter_40000.caffemodel \
  --imdb voc_2007_test \
  --cfg experiments/cfgs/piecewise.yml
