#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/svm_vgg16.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_svms.py --gpu $1 \
  --def models/VGG16/test.prototxt \
  --net output/default/voc_2007_trainval/vgg16_fast_rcnn_iter_40000.caffemodel \
  --imdb voc_2007_trainval \
  --cfg experiments/cfgs/svm.yml

time ./tools/test_net.py --gpu $1 \
  --def models/VGG16/test.prototxt \
  --net output/default/voc_2007_trainval/vgg16_fast_rcnn_iter_40000_svm.caffemodel \
  --imdb voc_2007_test \
  --cfg experiments/cfgs/svm.yml
