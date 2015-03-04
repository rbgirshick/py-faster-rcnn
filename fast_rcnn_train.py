#!/usr/bin/env python

import fast_rcnn_config as conf
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
import caffe
import finetuning
import datasets.pascal_voc
import bbox_regression_targets

from caffe.proto import caffe_pb2
import google.protobuf as pb2

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

class SolverWrapper(object):
    def __init__(self, solver_prototxt, pretrained_model=None):
        self.bbox_means = None
        self.bbox_stds = None

        self.solver = caffe.SGDSolver(solver_prototxt)
        if pretrained_model is not None:
            print 'Loading pretrained model weights from {:s}' \
                .format(pretrained_model)
            self.solver.net.copy_from(pretrained_model)

        self.solver_param = caffe_pb2.SolverParameter()
        with open(solver_prototxt, 'rt') as f:
            pb2.text_format.Merge(f.read(), self.solver_param)

    def snapshot(self):
        assert self.bbox_stds is not None
        assert self.bbox_means is not None

        stds = self.bbox_stds.ravel()[np.newaxis, np.newaxis, :, np.newaxis]
        means = self.bbox_means.ravel()[np.newaxis, np.newaxis, np.newaxis, :]

        # save original values
        orig_0 = self.solver.net.params['fc8_pascal_bbox'][0].data.copy()
        orig_1 = self.solver.net.params['fc8_pascal_bbox'][1].data.copy()

        # scale and shift with bbox reg unnormalization; then save snapshot
        self.solver.net.params['fc8_pascal_bbox'][0].data[...] = \
                self.solver.net.params['fc8_pascal_bbox'][0].data * stds
        self.solver.net.params['fc8_pascal_bbox'][1].data[...] = \
                self.solver.net.params['fc8_pascal_bbox'][1].data + means

        filename = self.solver_param.snapshot_prefix + \
              '_bbox06_iter_{:d}'.format(self.solver.iter) + '.caffemodel'
        self.solver.net.save(str(filename))
        print 'Wrote snapshot to: {:s}'.format(filename)

        # restore net to original state
        self.solver.net.params['fc8_pascal_bbox'][0].data[...] = orig_0
        self.solver.net.params['fc8_pascal_bbox'][1].data[...] = orig_1

# TODO(rbg): move into SolverWrapper
def train_model(sw, roidb, max_epochs=100):
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
            sw.solver.net.blobs['data'].reshape(base_shape[0], base_shape[1],
                                                base_shape[2], base_shape[3])
            sw.solver.net.blobs['rois'].reshape(num_rois, 5, 1, 1)
            sw.solver.net.blobs['labels'].reshape(num_rois, 1, 1, 1)
            sw.solver.net.blobs['bbox_targets'] \
                .reshape(num_rois, 4 * conf.NUM_CLASSES, 1, 1)
            sw.solver.net.blobs['bbox_loss_weights'] \
                .reshape(num_rois, 4 * conf.NUM_CLASSES, 1, 1)
            # Copy data into net's input blobs
            sw.solver.net.blobs['data'].data[...] = \
                im_blob.astype(np.float32, copy=False)
            sw.solver.net.blobs['rois'].data[...] = \
                rois_blob[:, :, np.newaxis, np.newaxis] \
                .astype(np.float32, copy=False)
            sw.solver.net.blobs['labels'].data[...] = \
                labels_blob[:, np.newaxis, np.newaxis, np.newaxis] \
                .astype(np.float32, copy=False)
            sw.solver.net.blobs['bbox_targets'].data[...] = \
                bbox_targets_blob[:, :, np.newaxis, np.newaxis] \
                .astype(np.float32, copy=False)
            sw.solver.net.blobs['bbox_loss_weights'].data[...] = \
                bbox_loss_weights_blob[:, :, np.newaxis, np.newaxis] \
                .astype(np.float32, copy=False)

            sw.solver.step(1)
            if sw.solver.iter % conf.SNAPSHOT_ITERS == 0:
                sw.snapshot()

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

    return roidb

if __name__ == '__main__':
    args = parse_args()

    # set up caffe
    caffe.set_phase_train()
    caffe.set_mode_gpu()
    if args.gpu_id is not None:
        caffe.set_device(args.gpu_id)

    imdb_train = datasets.pascal_voc('trainval', '2007')

    # enhance roidb to contain some useful derived quanties
    roidb_train = training_roidb(imdb_train)

    # enhance roidb to contain bounding-box regression targets
    means, stds = \
        bbox_regression_targets.append_bbox_regression_targets(roidb_train)

    # CAFFE_MODEL = '/data/reference_caffe_nets/ilsvrc_2012_train_iter_310k'
    # SOLVER_DEF = './model-defs/pyramid_solver.prototxt'
    CAFFE_MODEL = '/data/reference_caffe_nets/VGG_ILSVRC_16_layers.caffemodel'
    if args.solver is None:
        args.solver = './model-defs/vgg16_solver.prototxt'

    sw = SolverWrapper(args.solver, pretrained_model=CAFFE_MODEL)
    sw.bbox_means = means
    sw.bbox_stds = stds

    train_model(sw, roidb_train, max_epochs=args.epochs)
    sw.snapshot()
