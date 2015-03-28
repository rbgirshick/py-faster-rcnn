# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import numpy as np
import cv2
import matplotlib.pyplot as plt
import fast_rcnn_config as conf
import utils.blob

def get_minibatch(roidb):
    """
    Given a roidb, construct a minibatch sampled from it.
    """
    num_images = len(roidb)
    # Infer number of classes from the number of columns in gt_overlaps
    num_classes = roidb[0]['gt_overlaps'].shape[1]
    # Sample random scales to use for each image in this batch
    random_scale_inds = \
        np.random.randint(0, high=len(conf.SCALES), size=num_images)
    assert(conf.BATCH_SIZE % num_images == 0), \
        'num_images ({}) must divide BATCH_SIZE ({})'.format(num_images,
                                                             conf.BATCH_SIZE)
    rois_per_image = conf.BATCH_SIZE / num_images
    fg_rois_per_image = np.round(conf.FG_FRACTION * rois_per_image)

    # Get the input image blob, formatted for caffe
    im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)

    # Now, build the region of interest and label blobs
    rois_blob = np.zeros((0, 5), dtype=np.float32)
    labels_blob = np.zeros((0), dtype=np.float32)
    bbox_targets_blob = np.zeros((0, 4 * num_classes), dtype=np.float32)
    bbox_loss_blob = np.zeros(bbox_targets_blob.shape, dtype=np.float32)
    # all_overlaps = []
    for im_i in xrange(num_images):
        labels, overlaps, im_rois, bbox_targets, bbox_loss \
            = _sample_rois(roidb[im_i], fg_rois_per_image, rois_per_image)

        # Add to ROIs blob
        feat_rois = _map_im_rois_to_feat_rois(im_rois, im_scales[im_i])
        batch_ind = im_i * np.ones((feat_rois.shape[0], 1))
        rois_blob_this_image = np.hstack((batch_ind, feat_rois))
        rois_blob = np.vstack((rois_blob, rois_blob_this_image))

        # Add to labels, bbox targets, and bbox loss blobs
        labels_blob = np.hstack((labels_blob, labels))
        bbox_targets_blob = np.vstack((bbox_targets_blob, bbox_targets))
        bbox_loss_blob = np.vstack((bbox_loss_blob, bbox_loss))
        # all_overlaps = np.hstack((all_overlaps, overlaps))

    # For debug visualizations
    # _vis_minibatch(im_blob, rois_blob, labels_blob, all_overlaps)

    return im_blob, rois_blob, labels_blob, bbox_targets_blob, bbox_loss_blob

def _sample_rois(roidb, fg_rois_per_image, rois_per_image):
    """
    Generate a random sample of ROIs comprising foreground and background
    examples.
    """
    # label = class ROI has max overlap with
    labels = roidb['max_classes']
    overlaps = roidb['max_overlaps']
    rois = roidb['boxes']

    # Select foreground ROIs as those with >= FG_THRESH overlap
    fg_inds = np.where(overlaps >= conf.FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground ROIs
    fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_inds.size)
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = np.random.choice(fg_inds, size=fg_rois_per_this_image,
                                   replace=False)

    # Select background ROIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((overlaps < conf.BG_THRESH_HI) &
                       (overlaps >= conf.BG_THRESH_LO))[0]
    # Compute number of background ROIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = np.minimum(bg_rois_per_this_image,
                                        bg_inds.size)
    # Sample foreground regions without replacement
    if bg_inds.size > 0:
        bg_inds = np.random.choice(bg_inds, size=bg_rois_per_this_image,
                                   replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background ROIs to 0
    labels[fg_rois_per_this_image:] = 0
    overlaps = overlaps[keep_inds]
    rois = rois[keep_inds]

    # Infer number of classes from the number of columns in gt_overlaps
    num_classes = roidb['gt_overlaps'].shape[1]
    bbox_targets, bbox_loss_weights = \
            _get_bbox_regression_labels(roidb['bbox_targets'][keep_inds, :],
                                        num_classes)

    return labels, overlaps, rois, bbox_targets, bbox_loss_weights

def _get_image_blob(roidb, scale_inds):
    """
    Build an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    processed_ims = []
    im_scales = []
    for i in xrange(num_images):
        im = cv2.imread(roidb[i]['image'])
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        target_size = conf.SCALES[scale_inds[i]]
        im, im_scale = utils.blob.prep_im_for_blob(im, conf.PIXEL_MEANS,
                                                   target_size, conf.MAX_SIZE)
        im_scales.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = utils.blob.im_list_to_blob(processed_ims)

    return blob, im_scales

def _map_im_rois_to_feat_rois(im_rois, im_scale_factor):
    """
    Map a ROI in image-pixel coordinates to a ROI in feature coordinates.
    """
    feat_rois = np.round(im_rois * im_scale_factor / float(conf.FEAT_STRIDE))
    return feat_rois

def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """
    Bounding-box regression targets are stored in a compact form in the roidb.
    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets).
    The loss weights are similarly expanded.
    """
    # Return (N, K * 4, 1, 1) blob of regression targets
    # Return (N, K * 4, 1, 1) blob of loss weights
    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_loss_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind]
        start = 4 * cls
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_loss_weights[ind, start:end] = [1., 1., 1., 1.]
    return bbox_targets, bbox_loss_weights

def _vis_minibatch(im_blob, rois_blob, labels_blob, overlaps):
    num_images = im_blob.shape[0]
    for i in xrange(rois_blob.shape[0]):
      rois = rois_blob[i, :]
      im_ind = rois[0]
      roi = rois[1:] * conf.FEAT_STRIDE
      im = im_blob[im_ind, :, :, :].transpose((1, 2, 0)).copy()
      im += conf.PIXEL_MEANS
      im = im[:, :, (2, 1, 0)]
      im = im.astype(np.uint8)
      cls = labels_blob[i]
      plt.imshow(im)
      print 'class: ', cls, ' overlap: ', overlaps[i]
      plt.gca().add_patch(
          plt.Rectangle((roi[0], roi[1]), roi[2] - roi[0],
                        roi[3] - roi[1], fill=False,
                        edgecolor='r', linewidth=3)
          )
      plt.show()
