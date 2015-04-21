#! /bin/bash

set -x

export PYTHONUNBUFFERED="True"

# -----------------------------------------------------------------------------
LOG="experiments/logs/svd_caffenet.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec 3>&1 4>&2 &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/compress_net.py \
  --def models/CaffeNet/test.prototxt \
  --def-svd models/CaffeNet/compressed/test.prototxt \
  --net output/default/voc_2007_trainval/caffenet_fast_rcnn_iter_40000.caffemodel

time ./tools/test_net.py --gpu $1 \
  --def models/CaffeNet/compressed/test.prototxt \
  --net output/default/voc_2007_trainval/caffenet_fast_rcnn_iter_40000_svd_fc6_1024_fc7_256.caffemodel \
  --imdb voc_2007_test

# restore stdout/err
exec 1>&3 2>&4
# -----------------------------------------------------------------------------
