#! /bin/bash

set -x

export PYTHONUNBUFFERED="True"

# -----------------------------------------------------------------------------
LOG="experiments/logs/default_caffenet.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec 3>&1 4>&2 &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu $1 \
  --solver models/CaffeNet/solver.prototxt \
  --weights data/imagenet_models/CaffeNet.v2.caffemodel \
  --imdb voc_2007_trainval \
  --cfg experiments/cfgs/mb_size.yml

time ./tools/test_net.py --gpu $1 \
  --def models/CaffeNet/test.prototxt \
  --net output/default/voc_2007_trainval/caffenet_fast_rcnn_iter_40000.caffemodel \
  --imdb voc_2007_test \
  --cfg experiments/cfgs/mb_size.yml

# restore stdout/err
exec 1>&3 2>&4
# -----------------------------------------------------------------------------
