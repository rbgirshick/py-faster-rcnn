#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/default_vgg_cnn_m_1024.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu $1 \
  --solver models/VGG_CNN_M_1024/solver.prototxt \
  --weights data/imagenet_models/VGG_CNN_M_1024.v2.caffemodel \
  --imdb voc_2007_trainval

time ./tools/test_net.py --gpu $1 \
  --def models/VGG_CNN_M_1024/test.prototxt \
  --net output/default/voc_2007_trainval/vgg_cnn_m_1024_fast_rcnn_iter_40000.caffemodel \
  --imdb voc_2007_test
