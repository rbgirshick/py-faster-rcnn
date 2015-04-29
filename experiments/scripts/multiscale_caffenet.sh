#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/multiscale_caffenet.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu $1 \
  --solver models/CaffeNet/solver.prototxt \
  --weights data/imagenet_models/CaffeNet.v2.caffemodel \
  --imdb voc_2007_trainval \
  --cfg experiments/cfgs/multiscale.yml

time ./tools/test_net.py --gpu $1 \
  --def models/CaffeNet/test.prototxt \
  --net output/multiscale/voc_2007_trainval/caffenet_fast_rcnn_multiscale_iter_40000.caffemodel \
  --imdb voc_2007_test \
  --cfg experiments/cfgs/multiscale.yml
