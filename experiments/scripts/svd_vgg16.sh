#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/svd_vgg16.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/compress_net.py \
  --def models/VGG16/test.prototxt \
  --def-svd models/VGG16/compressed/test.prototxt \
  --net output/default/voc_2007_trainval/vgg16_fast_rcnn_iter_40000.caffemodel

time ./tools/test_net.py --gpu $1 \
  --def models/VGG16/compressed/test.prototxt \
  --net output/default/voc_2007_trainval/vgg16_fast_rcnn_iter_40000_svd_fc6_1024_fc7_256.caffemodel \
  --imdb voc_2007_test
