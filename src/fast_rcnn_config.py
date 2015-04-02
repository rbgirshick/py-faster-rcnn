# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

#
# README
#
# This file specifies default config options for Fast R-CNN. You should not
# change values in this file. Instead, you should write a config YAML file
# and use cfg_from_file(yaml_file) to load it and override the default options.
#
# - See tools/{train,test}_net.py for example code that uses cfg_from_file().
# - See examples/multiscale.yml for an example YAML config override file.
#

import os
import os.path as osp
import sys
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

# Add caffe to PYTHONPATH
caffe_path = osp.abspath(osp.join(osp.dirname(__file__), '..',
                                  'caffe-fast-rcnn', 'python'))
sys.path.insert(0, caffe_path)

__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

#
# Training options
#

__C.TRAIN                 = edict()

# Scales to use during training (can list multiple scales)
# Each scale is the pixel size of an image's shortest side
__C.TRAIN.SCALES          = (600,)

# Max pixel size of the longest side of a scaled input image
__C.TRAIN.MAX_SIZE        = 1000

# Images to use per minibatch
__C.TRAIN.IMS_PER_BATCH   = 2

# Minibatch size (number of regions of interest [ROIs])
__C.TRAIN.BATCH_SIZE      = 128

# Fraction of minibatch that is labeled foreground (i.e. class > 0)
__C.TRAIN.FG_FRACTION     = 0.25

# Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
__C.TRAIN.FG_THRESH       = 0.5

# Overlap threshold for a ROI to be considered background (class = 0 if
# overlap in [LO, HI))
__C.TRAIN.BG_THRESH_HI    = 0.5
__C.TRAIN.BG_THRESH_LO    = 0.1

# Use horizontally-flipped images during training?
__C.TRAIN.USE_FLIPPED     = True

# Overlap required between a ROI and ground-truth box in order for that ROI to
# be used as a bounding-box regression training example
__C.TRAIN.BBOX_THRESH     = 0.5

# Iterations between snapshots
__C.TRAIN.SNAPSHOT_ITERS  = 10000

# solver.prototxt specifies the snapshot path prefix, this adds an optional
# infix to yield the path: <prefix>[_<infix>]_iters_XYZ.caffemodel
__C.TRAIN.SNAPSHOT_INFIX  = ''

#
# Testing options
#

__C.TEST            = edict()

# Scales to use during testing (can list multiple scales)
# Each scale is the pixel size of an image's shortest side
__C.TEST.SCALES     = (600,)

# Max pixel size of the longest side of a scaled input image
__C.TEST.MAX_SIZE   = 1000

# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
__C.TEST.NMS        = 0.3

# Experimental: use binary logistic regression scores instead of K-way softmax
# scores when testing
__C.TEST.BINARY     = False

#
# MISC
#

# The mapping from image coordinates to feature map coordinates might cause
# some boxes that are distinct in image space to become identical in feature
# coordinates. If DEDUP_BOXES > 0, then DEDUP_BOXES is used as the scale factor
# for identifying duplicate boxes.
# 1/16 is correct for {Alex,Caffe}Net, VGG_CNN_M_1024, and VGG_16
__C.DEDUP_BOXES     = 1./16.

# Pixel mean values (BGR order) as a (1, 1, 3) array
# These are the values originally used for training VGG_16
__C.PIXEL_MEANS     = np.array([[[102.9801, 115.9465, 122.7717]]])

# For reproducibility
__C.RNG_SEED        = 3

# A small number that's used many times
__C.EPS             = 1e-14

# Root directory of project
__C.ROOT_DIR        = osp.abspath(osp.join(osp.dirname(__file__), '..'))

# Place outputs under an experiments directory
__C.EXP_DIR         = 'default'

def get_output_path(imdb, net):
    path = os.path.join(__C.ROOT_DIR, 'output', __C.EXP_DIR, imdb.name)
    if net is None:
        return path
    else:
        return os.path.join(path, net.name)

def _merge_a_into_b(a, b):
    """
    Merge config dictionary a into config dictionary b, clobbering the options
    in b whenever they are also specified in a.
    """
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
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v

def cfg_from_file(filename):
    """
    Load a config file and merge it into the default options specified in this
    file.
    """
    import yaml
    global __C
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)
