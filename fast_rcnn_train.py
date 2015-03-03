#!/usr/bin/env python

import sys
caffe_path = '../caffe/python'
sys.path.insert(0, caffe_path)

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
import caffe
import finetuning
import fast_rcnn_config as conf
import datasets.pascal_voc
import bbox_regression_targets

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

def load_solver(solver_def_path, pretrained_model=None):
    solver = caffe.SGDSolver(solver_def_path)
    if pretrained_model is not None:
        print 'Loading pretrained model weights from {:s}' \
            .format(pretrained_model)
        solver.net.copy_from(pretrained_model)
    return solver

def train_model_random_scales(solver_def_path, roidb,
                              pretrained_model=None, GPU_ID=None,
                              max_epochs=100):
    caffe.set_phase_train()
    caffe.set_mode_gpu()
    if GPU_ID is not None:
        caffe.set_device(GPU_ID)

    solver = load_solver(solver_def_path, pretrained_model=pretrained_model)

    for epoch in xrange(max_epochs):
        shuffled_inds = np.random.permutation(np.arange(len(roidb)))
        lim = (len(shuffled_inds) / conf.IMS_PER_BATCH) * conf.IMS_PER_BATCH
        shuffled_inds = shuffled_inds[0:lim]
        for shuffled_i in xrange(0, len(shuffled_inds), conf.IMS_PER_BATCH):
            # start_t = time.time()
            db_inds = shuffled_inds[shuffled_i:shuffled_i + conf.IMS_PER_BATCH]
            minibatch_db = [roidb[i] for i in db_inds]
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

            solver.step(1)
    return solver

def training_roidb(imdb):
    """
    Enriched the imdb's roidb by adding some derived quantities that
    are useful for training. This function precomputes the maximum
    overlap, taken over ground-truth boxes, between each ROI and
    each ground-truth box. The class with maximum overlap is also
    recorded.
    """
    roidb = imdb.roidb
    for i in xrange(len(imdb.image_index)):
        roidb[i]['image'] = imdb.image_path_at(i)
        # need gt_overlaps as a dense array for argmax
        gt_overlaps = roidb[i]['gt_overlaps'].toarray()
        # max overlap with gt over classes (columns)
        max_overlaps = gt_overlaps.max(axis=1)
        # gt class that had the max overlap
        max_classes = gt_overlaps.argmax(axis=1)
        roidb[i]['max_classes'] = max_classes
        roidb[i]['max_overlaps'] = max_overlaps
        # sanity checks
        # max overlap of 0 => class should be zero (background)
        zero_inds = np.where(max_overlaps == 0)[0]
        assert all(max_classes[zero_inds] == 0)
        # max overlap > 0 => class should not be zero (must be a fg class)
        nonzero_inds = np.where(max_overlaps > 0)[0]
        assert all(max_classes[nonzero_inds] != 0)

#    import cPickle
#    import gzip
#    WINDOW_DB = './data/window_file_voc_2007_trainval.txt.pz'
#    with gzip.GzipFile(WINDOW_DB, 'rb') as f:
#        windb = cPickle.load(f)
#
#    from IPython import embed; embed()

    return roidb

if __name__ == '__main__':
    args = parse_args()

    # CAFFE_MODEL = '/data/reference_caffe_nets/ilsvrc_2012_train_iter_310k'
    # SOLVER_DEF = './model-defs/pyramid_solver.prototxt'

    imdb_train = datasets.pascal_voc('trainval', '2007')

    # enhance roidb to contain some useful derived quanties
    roidb_train = training_roidb(imdb_train)

    # TODO(rbg): need to save means and stds
    # further enhance roidb to contain bounding-box regression targets
    means, stds = \
        bbox_regression_targets.append_bbox_regression_targets(roidb_train)

    CAFFE_MODEL = '/data/reference_caffe_nets/VGG_ILSVRC_16_layers.caffemodel'
    if args.solver is None:
        args.solver = './model-defs/vgg16_solver.prototxt'

    solver = train_model_random_scales(args.solver, roidb_train,
                                       pretrained_model=CAFFE_MODEL,
                                       GPU_ID=args.gpu_id,
                                       max_epochs=args.epochs)
