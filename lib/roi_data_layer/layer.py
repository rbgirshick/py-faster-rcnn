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
from multiprocessing import Process, queues

class RoIDataLayer(caffe.Layer):
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

    @staticmethod
    def _prefetch(minibatch_db, num_classes, output_queue):
        blobs = get_minibatch(minibatch_db, num_classes)
        output_queue.put(blobs)

    def _get_next_minibatch(self):
        db_inds = self._get_next_minibatch_inds()
        minibatch_db = [self._roidb[i] for i in db_inds]
        if cfg.TRAIN.USE_PREFETCH:
            self._prefetch_process = Process(target=RoIDataLayer._prefetch,
                                             args=(minibatch_db,
                                                   self._num_classes,
                                                   self._prefetch_queue))
            self._prefetch_process.start()
        else:
            return get_minibatch(minibatch_db, self._num_classes)

    def set_roidb(self, roidb):
        self._roidb = roidb
        self._shuffle_roidb_inds()
        if cfg.TRAIN.USE_PREFETCH:
            self._get_next_minibatch()

    def setup(self, bottom, top):
        if cfg.TRAIN.USE_PREFETCH:
            self._prefetch_process = None
            self._prefetch_queue = queues.SimpleQueue()

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

    def forward(self, bottom, top):
        if cfg.TRAIN.USE_PREFETCH:
            blobs = self._prefetch_queue.get()
            self._get_next_minibatch()
        else:
            blobs = self._get_next_minibatch()

        for blob_name, blob in blobs.iteritems():
            top_ind = self._name_to_top_map[blob_name]
            # Reshape net's input blobs
            top[top_ind].reshape(*(blob.shape))
            # Copy data into net's input blobs
            top[top_ind].data[...] = blob.astype(np.float32, copy=False)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
