#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/no_bbox_reg_vgg16.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu $1 \
  --solver models/VGG16/no_bbox_reg/solver.prototxt \
  --weights data/imagenet_models/VGG16.v2.caffemodel \
  --imdb voc_2007_trainval \
  --cfg experiments/cfgs/no_bbox_reg.yml

time ./tools/test_net.py --gpu $1 \
  --def models/VGG16/no_bbox_reg/test.prototxt \
  --net output/no_bbox_reg/voc_2007_trainval/vgg16_fast_rcnn_no_bbox_reg_iter_40000.caffemodel \
  --imdb voc_2007_test \
  --cfg experiments/cfgs/no_bbox_reg.yml
