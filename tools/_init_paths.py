# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os.path as osp
import sys

this_dir = osp.dirname(__file__)

# Add caffe to PYTHONPATH
caffe_path = osp.join(this_dir, '..', 'caffe-fast-rcnn', 'python')

if caffe_path not in sys.path:
  sys.path.insert(0, caffe_path)

lib_path = osp.join(this_dir, '..', 'lib')

if lib_path not in sys.path:
  sys.path.insert(0, lib_path)
