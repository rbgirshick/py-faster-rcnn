#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Reval = re-eval. Re-evaluate saved detections."""

import _init_paths
from fast_rcnn.test import apply_nms
from fast_rcnn.config import cfg
from datasets.factory import get_imdb
import cPickle
import os, sys
import numpy as np

def from_mats(imdb_name, output_dir):
    import scipy.io as sio

    imdb = get_imdb(imdb_name)
    aps = []
    for i, cls in enumerate(imdb.classes[1:]):
        mat = sio.loadmat(os.path.join(output_dir, cls + '_pr.mat'))
        ap = mat['ap'][0, 0] * 100
        apAuC = mat['ap_auc'][0, 0] * 100
        print '!!! {} : {:.1f} {:.1f}'.format(cls, ap, apAuC)
        aps.append(ap)

    print '~~~~~~~~~~~~~~~~~~~'
    print 'Results (from mat files):'
    for ap in aps:
        print '{:.1f}'.format(ap)
    print '{:.1f}'.format(np.array(aps).mean())
    print '~~~~~~~~~~~~~~~~~~~'


def from_dets(imdb_name, output_dir):
    imdb = get_imdb(imdb_name)
    imdb.config['use_salt'] = False
    imdb.config['cleanup'] = False
    with open(os.path.join(output_dir, 'detections.pkl'), 'rb') as f:
        dets = cPickle.load(f)

    print 'Applying NMS to all detections'
    nms_dets = apply_nms(dets, cfg.TEST.NMS)

    print 'Evaluating detections'
    imdb.evaluate_detections(nms_dets, output_dir)

if __name__ == '__main__':
    # 'output/top_1000/voc_2007_test/vgg_cnn_m_1024_fast_rcnn_iter_40000'
    output_dir = sys.argv[1]
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                 '..', output_dir))
    imdb_name = 'voc_2007_test'

    if len(sys.argv) > 2:
        from_mats(imdb_name, output_dir)
    else:
        from_dets(imdb_name, output_dir)
