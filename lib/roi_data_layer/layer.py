# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import caffe
from fast_rcnn.config import cfg
from roi_data_layer.minibatch import get_minibatch
import numpy as np
import yaml

class DataLayer(caffe.Layer):
    """Fast R-CNN data layer."""

    def _shuffle_roidb_inds(self):
        self._perm = np.random.permutation(np.arange(len(self._roidb)))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._roidb):
            self._shuffle_roidb_inds()

        db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
        self._cur += cfg.TRAIN.IMS_PER_BATCH
        return db_inds

    def _set_next_minibatch(self):
        db_inds = self._get_next_minibatch_inds()
        minibatch_db = [self._roidb[i] for i in db_inds]
        self._blobs = get_minibatch(minibatch_db, self._num_classes)

    def set_roidb(self, roidb):
        self._roidb = roidb
        self._shuffle_roidb_inds()

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str_)

        self._num_classes = layer_params['num_classes']

        self._name_to_top_map = {
            'data': 0,
            'rois': 1,
            'labels': 2,
            'bbox_targets': 3,
            'bbox_loss_weights': 4}

        # data
        top[0].reshape(1, 3, 1, 1)
        # rois
        top[1].reshape(1, 5)
        # labels
        top[2].reshape(1)
        # bbox_targets
        top[3].reshape(1, self._num_classes * 4)
        # bbox_loss_weights
        top[4].reshape(1, self._num_classes * 4)

        # TODO(rbg):
        # Start a prefetch thread that calls self._get_next_minibatch()

    def forward(self, bottom, top):
        # TODO(rbg):
        # wait for prefetch thread to finish
        self._set_next_minibatch()

        for blob_name, blob in self._blobs.iteritems():
            top_ind = self._name_to_top_map[blob_name]
            # Reshape net's input blobs
            top[top_ind].reshape(*(blob.shape))
            # Copy data into net's input blobs
            top[top_ind].data[...] = blob.astype(np.float32, copy=False)

        # TODO(rbg):
        # start next prefetch thread

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
