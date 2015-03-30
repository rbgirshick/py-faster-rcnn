# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
import sys
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

# Add caffe to PYTHONPATH
caffe_path = os.path.abspath(os.path.join('..', 'caffe-master', 'python'))
sys.path.insert(0, caffe_path)

__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

#
# Training
#

__C.TRAIN                 = edict()

# SCALES          = (480, 576, 688, 864, 1200)
__C.TRAIN.SCALES          = (600,)

# Max pixel size of a scaled input image
__C.TRAIN.MAX_SIZE        = 1000

# Images per batch
__C.TRAIN.IMS_PER_BATCH   = 2

# Minibatch size
__C.TRAIN.BATCH_SIZE      = 128

# Fraction of minibatch that is foreground labeled (class > 0)
__C.TRAIN.FG_FRACTION     = 0.25

# Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
__C.TRAIN.FG_THRESH       = 0.5

# Overlap threshold for a ROI to be considered background (class = 0 if
# overlap in [0.1, 0.5))
__C.TRAIN.BG_THRESH_HI    = 0.5
__C.TRAIN.BG_THRESH_LO    = 0.1

# Use horizontally-flipped images during training?
__C.TRAIN.USE_FLIPPED     = True

__C.TRAIN.BBOX_THRESH     = 0.5
__C.TRAIN.SNAPSHOT_ITERS  = 10000
__C.TRAIN.SNAPSHOT_INFIX  = ''

#
# Testing
#

__C.TEST            = edict()
__C.TEST.SCALES     = (600,)
__C.TEST.MAX_SIZE   = 1000
__C.TEST.NMS        = 0.3
__C.TEST.BINARY     = False

#
# MISC
#

# Pixel mean values (BGR order) as a (1, 1, 3) array
# These are the values originally used for training VGG_16
__C.PIXEL_MEANS     = np.array([[[102.9801, 115.9465, 122.7717]]])

# Stride in input image pixels at ROI pooling level (network specific)
# 16 is true for {Alex,Caffe}Net, VGG_CNN_M_1024, and VGG16
__C.FEAT_STRIDE     = 16

# For reproducibility [currently the caffe seed isn't fixed...so there's that]
__C.RNG_SEED        = 3

# A small number that's used many times
__C.EPS             = 1e-14

def merge_a_into_b(a, b):
    if type(a) is not edict:
        return
    for k, v in a.iteritems():
        # a must specify keys that are in b
        if not b.has_key(k):
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        if type(b[k]) is not type(v):
            raise ValueError('Type mismatch ({} vs. {}) for config key: {}'.
                              format(type(b[k]), type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v

def cfg_from_file(filename):
    import yaml
    global __C
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    merge_a_into_b(yaml_cfg, __C)
