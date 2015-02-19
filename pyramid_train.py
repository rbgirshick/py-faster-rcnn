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

from deep_pyramid import DeepPyramid
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
from keyboard import keyboard

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

def train_model(solver_def_path, window_db_path, pretrained_model=None,
                GPU_ID=None):
    solver, window_db = \
        load_solver_and_window_db(solver_def_path,
                                  window_db_path,
                                  pretrained_model=pretrained_model)
    caffe.set_phase_train()
    caffe.set_mode_gpu()
    if GPU_ID is not None:
        caffe.set_device(GPU_ID)

    max_epochs = 100
    dp = DeepPyramid(solver.net)
    for epoch in xrange(max_epochs):
        # TODO(rbg): shuffle window_db
        for db_i in xrange(len(window_db)):
            start_t = time.time()

            # Load image and prepare pyramid input batch
            im = cv2.imread(window_db[db_i]['image'])
            im_pyra, pyra_scales = dp.get_image_pyramid(im)
            im_pyra_batch = dp.image_pyramid_to_batch(im_pyra)

            # Load boxes and convert to pyramid ROIs
            im_rois = window_db[db_i]['windows'][:, 2:]
            pyra_levels, pyra_rois = \
                dp.im_to_feat_pyramid_coords(im_rois, pyra_scales)
            pyra_levels_and_rois = \
                np.append(pyra_levels[:, np.newaxis], pyra_rois, axis=1)
            # TODO(rbg): remove duplicate pyra_rois (due to roi aliasing)
            # map (class, overlap) to hard label
            labels = window_db[db_i]['windows'][:, 0]
            overlaps = window_db[db_i]['windows'][:, 1]
            # Label all ROIs with < 0.5 overlap as background (label = 0)
            fg_inds = np.where(overlaps >= 0.5)
            bg_inds = np.where(overlaps < 0.5)
            labels[bg_inds] = 0

            # Take a random sample of the examples
            bg_inds = np.asarray(bg_inds).T
            fg_inds = np.asarray(fg_inds).T
            np.random.shuffle(bg_inds)
            num_to_sample = 1500 - fg_inds.size
            keep_inds = np.append(fg_inds,
                                  bg_inds[0:num_to_sample], axis=0).ravel()
            pyra_levels_and_rois = pyra_levels_and_rois[keep_inds]
            labels = labels[keep_inds]
            im_rois = im_rois[keep_inds]

#            for c in xrange(1, 21):
#                plt.imshow(im)
#                c_inds = np.where(labels == c)[0]
#                if c_inds.size == 0:
#                    continue
#                print 'class: ', c
#                for fg_i in c_inds:
#                    roi = im_rois[fg_i, :]
#                    plt.gca().add_patch(
#                        plt.Rectangle((roi[0], roi[1]), roi[2] - roi[0],
#                                      roi[3] - roi[1], fill=False,
#                                      edgecolor='r', linewidth=3)
#                        )
#                plt.show()

            # Reshape net's input blobs
            base_shape = im_pyra[0].shape
            num_rois = pyra_levels_and_rois.shape[0]
            solver.net.blobs['data'].reshape(dp.num_levels, base_shape[2],
                                             base_shape[0], base_shape[1])
            solver.net.blobs['rois'].reshape(num_rois, 5, 1, 1)
            solver.net.blobs['labels'].reshape(num_rois, 1, 1, 1)
            # Copy data into net's input blobs
            solver.net.blobs['data'].data[...] = im_pyra_batch
            solver.net.blobs['rois'].data[...] = \
                pyra_levels_and_rois[:, :, np.newaxis, np.newaxis]
            solver.net.blobs['labels'].data[...] = \
                labels[:, np.newaxis, np.newaxis, np.newaxis]

            print 'epoch {:d} image {:d}'.format(epoch, db_i)
            print_label_stats(labels)
            solver.step(1)
            # Periodically snapshot and test
            print 'Elapsed time: {:.4f}'.format(time.time() - start_t)

def train_model_random_scales(solver_def_path, window_db_path,
                              pretrained_model=None, GPU_ID=None):
    solver, window_db = \
        load_solver_and_window_db(solver_def_path,
                                  window_db_path,
                                  pretrained_model=pretrained_model)
    caffe.set_phase_train()
    caffe.set_mode_gpu()
    if GPU_ID is not None:
        caffe.set_device(GPU_ID)

    max_epochs = 100
    for epoch in xrange(max_epochs):
        shuffled_inds = np.random.permutation(np.arange(len(window_db)))
        lim = (len(shuffled_inds) / conf.IMS_PER_BATCH) * conf.IMS_PER_BATCH
        shuffled_inds = shuffled_inds[0:lim]
        for shuffled_i in xrange(0, len(shuffled_inds), conf.IMS_PER_BATCH):
            start_t = time.time()
            db_inds = shuffled_inds[shuffled_i:shuffled_i + conf.IMS_PER_BATCH]
            minibatch_db = [window_db[i] for i in db_inds]
            im_blob, rois_blob, labels_blob = \
                finetuning.get_minibatch(minibatch_db)

            # Reshape net's input blobs
            base_shape = im_blob.shape
            num_rois = rois_blob.shape[0]
            solver.net.blobs['data'].reshape(base_shape[0], base_shape[1],
                                             base_shape[2], base_shape[3])
            solver.net.blobs['rois'].reshape(num_rois, 5, 1, 1)
            solver.net.blobs['labels'].reshape(num_rois, 1, 1, 1)
            # Copy data into net's input blobs
            solver.net.blobs['data'].data[...] = \
                im_blob.astype(np.float32, copy=False)
            solver.net.blobs['rois'].data[...] = \
                rois_blob[:, :, np.newaxis, np.newaxis] \
                .astype(np.float32, copy=False)
            solver.net.blobs['labels'].data[...] = \
                labels_blob[:, np.newaxis, np.newaxis, np.newaxis] \
                .astype(np.float32, copy=False)

            # print 'epoch {:d} image {:d}'.format(epoch, db_i)
            # print_label_stats(labels_blob)
            solver.step(1)
            # Periodically snapshot and test
            # print 'Elapsed time: {:.4f}'.format(time.time() - start_t)


if __name__ == '__main__':
    # CAFFE_MODEL = '/data/reference_caffe_nets/ilsvrc_2012_train_iter_310k'
    # SOLVER_DEF = './model-defs/pyramid_solver.prototxt'
    CAFFE_MODEL = '/data/reference_caffe_nets/VGG_ILSVRC_16_layers.caffemodel'
    SOLVER_DEF = './model-defs/vgg16_solver.prototxt'
    # CAFFE_MODEL = '/data/reference_caffe_nets/bvlc_googlenet.caffemodel'
    # SOLVER_DEF = './model-defs/googlenet_solver.prototxt'
    WINDOW_DB = './data/window_file_voc_2007_trainval.txt.pz'
    GPU_ID = 0 if len(sys.argv) == 1 else int(sys.argv[1])
    train_model_random_scales(SOLVER_DEF, WINDOW_DB,
                              pretrained_model=CAFFE_MODEL,
                              GPU_ID=GPU_ID)
