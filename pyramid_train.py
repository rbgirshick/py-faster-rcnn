#!/usr/bin/env python

# Define train prototxt with two inputs:
#  0: pyramid batch
#  1: ROI inputs
# Loop:
#   get next window_db entry
#   load image with cv2
#   use deep pyramid library to construct input image batch (pyramid)
#   use deep pyramid library to maps image space ROIs into pyramid ROIs
#   use deep pyramid library to compute label for each ROI
#   prefill solver's net with two input blobs
#   execute solver step(1)

import argparse
from deep_pyramid import DeepPyramid
import bbox_regression_targets
import sys
caffe_path = '../caffe/python'
sys.path.insert(0, caffe_path)
import caffe
import numpy as np
import cPickle
import gzip
import cv2
import time
import matplotlib.pyplot as plt
import finetuning
import fast_rcnn_config as conf

def parse_args():
    """
    Parse input arguments
    """

    parser = argparse.ArgumentParser(description='Train a fast R-CNN')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--solver', dest='solver', help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--epochs', dest='epochs',
                        help='number of epoch to train',
                        default=16, type=int)

    args = parser.parse_args()
    return args

def print_label_stats(labels):
    counts = np.bincount(labels.astype(np.int))
    for i in xrange(len(counts)):
        if counts[i] > 0:
            print '{:3d}: {:4d}'.format(i, counts[i])

def load_window_db(path):
    with gzip.GzipFile(path, 'rb') as f:
        return cPickle.load(f)

def load_solver_and_window_db(solver_def_path, window_db_path,
                              pretrained_model=None):
    solver = caffe.SGDSolver(solver_def_path)
    if pretrained_model is not None:
        print 'Loading pretrained model weights from {:s}' \
            .format(pretrained_model)
        solver.net.copy_from(pretrained_model)
    print 'Loading window db from {:s}'.format(window_db_path)
    window_db = load_window_db(window_db_path)
    return solver, window_db

def train_model_random_scales(solver_def_path, window_db_path,
                              pretrained_model=None, GPU_ID=None,
                              max_epochs=100):
    solver, window_db = \
        load_solver_and_window_db(solver_def_path,
                                  window_db_path,
                                  pretrained_model=pretrained_model)
    means, stds = \
        bbox_regression_targets.append_bbox_regression_targets(window_db)

    caffe.set_phase_train()
    caffe.set_mode_gpu()
    if GPU_ID is not None:
        caffe.set_device(GPU_ID)

    for epoch in xrange(max_epochs):
        shuffled_inds = np.random.permutation(np.arange(len(window_db)))
        lim = (len(shuffled_inds) / conf.IMS_PER_BATCH) * conf.IMS_PER_BATCH
        shuffled_inds = shuffled_inds[0:lim]
        for shuffled_i in xrange(0, len(shuffled_inds), conf.IMS_PER_BATCH):
            start_t = time.time()
            db_inds = shuffled_inds[shuffled_i:shuffled_i + conf.IMS_PER_BATCH]
            minibatch_db = [window_db[i] for i in db_inds]
            im_blob, rois_blob, labels_blob, \
                bbox_targets_blob, bbox_loss_weights_blob = \
                    finetuning.get_minibatch(minibatch_db)

            # Reshape net's input blobs
            base_shape = im_blob.shape
            num_rois = rois_blob.shape[0]
            solver.net.blobs['data'].reshape(base_shape[0], base_shape[1],
                                             base_shape[2], base_shape[3])
            solver.net.blobs['rois'].reshape(num_rois, 5, 1, 1)
            solver.net.blobs['labels'].reshape(num_rois, 1, 1, 1)
            solver.net.blobs['bbox_targets'] \
                .reshape(num_rois, 4 * conf.NUM_CLASSES, 1, 1)
            solver.net.blobs['bbox_loss_weights'] \
                .reshape(num_rois, 4 * conf.NUM_CLASSES, 1, 1)
            # Copy data into net's input blobs
            solver.net.blobs['data'].data[...] = \
                im_blob.astype(np.float32, copy=False)
            solver.net.blobs['rois'].data[...] = \
                rois_blob[:, :, np.newaxis, np.newaxis] \
                .astype(np.float32, copy=False)
            solver.net.blobs['labels'].data[...] = \
                labels_blob[:, np.newaxis, np.newaxis, np.newaxis] \
                .astype(np.float32, copy=False)
            solver.net.blobs['bbox_targets'].data[...] = \
                bbox_targets_blob[:, :, np.newaxis, np.newaxis] \
                .astype(np.float32, copy=False)
            solver.net.blobs['bbox_loss_weights'].data[...] = \
                bbox_loss_weights_blob[:, :, np.newaxis, np.newaxis] \
                .astype(np.float32, copy=False)

            # print 'epoch {:d} image {:d}'.format(epoch, db_i)
            # print_label_stats(labels_blob)
            solver.step(1)
            # Periodically snapshot and test
            # print 'Elapsed time: {:.4f}'.format(time.time() - start_t)
    return solver


if __name__ == '__main__':
    args = parse_args()

    # CAFFE_MODEL = '/data/reference_caffe_nets/ilsvrc_2012_train_iter_310k'
    # SOLVER_DEF = './model-defs/pyramid_solver.prototxt'
    WINDOW_DB = './data/window_file_voc_2007_trainval.txt.pz'

    CAFFE_MODEL = '/data/reference_caffe_nets/VGG_ILSVRC_16_layers.caffemodel'
    if args.solver is None:
        args.solver = './model-defs/vgg16_solver.prototxt'

    solver = train_model_random_scales(args.solver, WINDOW_DB,
                                       pretrained_model=CAFFE_MODEL,
                                       GPU_ID=args.gpu_id,
                                       max_epochs=args.epochs)
