import numpy as np
import cv2
import matplotlib.pyplot as plt
import fast_rcnn_config as conf
from keyboard import keyboard

def sample_rois(labels, overlaps, rois, fg_rois_per_image, rois_per_image):
    """Generate a random sample of ROIs comprising foreground and background
    examples.

    Args:
      labels (1d np array): class labels from a window db
      overlaps (1d np array): class max overlap from a window db
      rois (2d np array): regions of interest in image coordinates, one per row
      fg_rois_per_image (int): target number of foreground ROIs
      rois_per_image (int): target number of ROIs to return in total

    Returns:
      labels (1d np array)
      overlaps (1d np array)
      rois (2d np array)
    """
    # Select foreground ROIs
    fg_inds = np.where(overlaps >= conf.FG_THRESH)[0]
    fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_inds.size)
    fg_inds = np.random.choice(fg_inds, size=fg_rois_per_this_image,
                               replace=False)
    # Select background ROIs
    bg_inds = np.where((overlaps < conf.BG_THRESH_HI) &
                       (overlaps >= conf.BG_THRESH_LO))[0]
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = np.minimum(bg_rois_per_this_image,
                                        bg_inds.size)
    bg_inds = np.random.choice(bg_inds, size=bg_rois_per_this_image,
                               replace=False)
    keep_inds = np.append(fg_inds, bg_inds)
    labels[bg_inds] = 0
    labels = labels[keep_inds]
    overlaps = overlaps[keep_inds]
    rois = rois[keep_inds]
    return labels, overlaps, rois

def get_image_blob(window_db, scale_inds, do_flip):
    """Build an input blob from the images in the window db at the specified
    scales.
    """
    num_images = len(window_db)
    max_shape = (0, 0, 0)
    processed_ims = []
    im_scale_factors = []
    for i in xrange(num_images):
        im = cv2.imread(window_db[i]['image'])
        if do_flip:
            im = im[:, ::-1, :]
        im = im.astype(np.float32, copy=False)
        im -= conf.PIXEL_MEANS
        im_shape = im.shape
        im_size = np.min(im_shape[0:2])
        im_size_big = np.max(im_shape[0:2])
        target_size = conf.SCALES[scale_inds[i]]
        im_scale = float(target_size) / float(im_size)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_big) > conf.MAX_SIZE:
            im_scale = float(conf.MAX_SIZE) / float(im_size_big)
        im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)
        max_shape = np.maximum(max_shape, im.shape)

    blob = np.zeros((num_images, max_shape[0],
                     max_shape[1], max_shape[2]), dtype=np.float32)
    for i in xrange(num_images):
        im = processed_ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
    # Move channels (axis 3) to axis 1
    # Axis order will become: (batch elem, channel, height, width)
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)
    return blob, im_scale_factors

def map_im_rois_to_feat_rois(im_rois, im_scale_factor):
    feat_rois = np.round(im_rois * im_scale_factor / conf.FEAT_STRIDE)
    return feat_rois

def get_minibatch(window_db, random_flip=False):
    # Decide to flip the entire batch or not
    do_flip = False if not random_flip else bool(np.random.randint(0, high=2))
    assert(not do_flip)
    num_images = len(window_db)
    # Sample random scales to use for each image in this batch
    random_scale_inds = \
        np.random.randint(0, high=len(conf.SCALES), size=num_images)
    assert(conf.BATCH_SIZE % num_images == 0), \
        'num_images must divide BATCH_SIZE'
    rois_per_image = conf.BATCH_SIZE / num_images
    fg_rois_per_image = np.round(conf.FG_FRACTION * rois_per_image)
    # Get the input blob, formatted for caffe
    # Takes care of random scaling and flipping
    im_blob, im_scale_factors = get_image_blob(window_db,
                                               random_scale_inds, do_flip)
    # Now, build the region of interest and label blobs
    rois_blob = np.zeros((0, 5), dtype=np.float32)
    labels_blob = np.zeros((0), dtype=np.float32)
    all_overlaps = []
    for im_i in xrange(num_images):
        # (labels, overlaps, x1, y1, x2, y2)
        labels = window_db[im_i]['windows'][:, 0]
        overlaps = window_db[im_i]['windows'][:, 1]
        im_rois = window_db[im_i]['windows'][:, 2:]
        if do_flip:
            im_rois[:, (0, 2)] = window_db[im_i]['width'] - \
                                 im_rois[:, (2, 0)] - 1
        labels, overlaps, im_rois = sample_rois(labels, overlaps, im_rois,
                                                fg_rois_per_image,
                                                rois_per_image)
        feat_rois = map_im_rois_to_feat_rois(im_rois, im_scale_factors[im_i])
        # Assert various bounds
        assert((feat_rois[:, 2] >= feat_rois[:, 0]).all())
        assert((feat_rois[:, 3] >= feat_rois[:, 1]).all())
        assert((feat_rois >= 0).all())
        assert((feat_rois < np.max(im_blob.shape[2:4]) *
                            im_scale_factors[im_i] / conf.FEAT_STRIDE).all())
        rois_blob_this_image = \
            np.append(im_i * np.ones((feat_rois.shape[0], 1)), feat_rois,
                      axis=1)
        rois_blob = np.append(rois_blob, rois_blob_this_image, axis=0)
        labels_blob = np.append(labels_blob, labels, axis=0)
        all_overlaps = np.append(all_overlaps, overlaps, axis=0)
    # vis_minibatch(im_blob, rois_blob, labels_blob, all_overlaps)
    return im_blob, rois_blob, labels_blob

def vis_minibatch(im_blob, rois_blob, labels_blob, overlaps):
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
