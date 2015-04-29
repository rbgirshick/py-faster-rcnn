#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/fc_only_vgg16.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu $1 \
  --solver models/VGG16/fc_only/solver.prototxt \
  --weights data/imagenet_models/VGG16.v2.caffemodel \
  --imdb voc_2007_trainval \
  --cfg experiments/cfgs/fc_only.yml

time ./tools/test_net.py --gpu $1 \
  --def models/VGG16/test.prototxt \
  --net output/fc_only/voc_2007_trainval/vgg16_fast_rcnn_fc_only_iter_40000.caffemodel \
  --imdb voc_2007_test \
  --cfg experiments/cfgs/fc_only.yml
