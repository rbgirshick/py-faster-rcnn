#!/usr/bin/env python

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '..', 'src')))

from datasets.factory import get_imdb
import cPickle
import fast_rcnn_test
from fast_rcnn_config import cfg

def main(imdb_name, output_dir):
    imdb = get_imdb(imdb_name)
    imdb.config['use_salt'] = False
    imdb.config['cleanup'] = False
    with open(os.path.join(output_dir, 'detections.pkl'), 'rb') as f:
        dets = cPickle.load(f)

    print 'Applying NMS to all detections'
    nms_dets = fast_rcnn_test.apply_nms(dets, cfg.TEST.NMS)

    print 'Evaluating detections'
    imdb.evaluate_detections(nms_dets, output_dir)

if __name__ == '__main__':
    # 'output/top_1000/voc_2007_test/vgg_cnn_m_1024_fast_rcnn_iter_40000'
    output_dir = sys.argv[1]
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                 '..', output_dir))
    imdb_name = 'voc_2007_test'
    main(imdb_name, output_dir)
