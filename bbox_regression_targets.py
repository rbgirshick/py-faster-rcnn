import numpy as np
import fast_rcnn_config as conf
import utils.cython_bbox

# window_db[im_i]['image']
# window_db[im_i]['rois']
# (labels, overlaps, x1, y1, x2, y2)
# labels = window_db[im_i]['windows'][:, 0]
# overlaps = window_db[im_i]['windows'][:, 1]
# im_rois = window_db[im_i]['windows'][:, 2:]

def compute_bbox_regression_targets(windows):
    labels = windows[:, 0]
    overlaps = windows[:, 1]
    rois = windows[:, 2:]

    # Indices of ground-truth ROIs
    gt_inds = np.where(overlaps == 1)[0]
    # Indices of examples for which we try to make predictions
    ex_inds = np.where(overlaps >= conf.BBOX_THRESH)[0]

    # Get IoU overlap between each ex ROI and gt ROI
    ex_gt_overlaps = utils.cython_bbox.bbox_overlaps(rois[ex_inds, :],
                                                     rois[gt_inds, :])

    gt_assignment = ex_gt_overlaps.argmax(axis=1)
    gt_rois = rois[gt_inds[gt_assignment], :]
    ex_rois = rois[ex_inds, :]

    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + conf.EPS
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + conf.EPS
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + conf.EPS
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + conf.EPS
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.zeros((windows.shape[0], 5), dtype=np.float32)
    targets[ex_inds, 0] = labels[ex_inds]
    targets[ex_inds, 1] = targets_dx
    targets[ex_inds, 2] = targets_dy
    targets[ex_inds, 3] = targets_dw
    targets[ex_inds, 4] = targets_dh
    return targets

def append_bbox_regression_targets(window_db):
    # For each image
    #   For each window with overlap >= FG_THRESH
    #       find the corresponding GT bounding box
    #       compute the regression targets between the two
    #
    # Compute mean and std of each (class, target) pair
    # Normalize targets
    # Need to save these values

    num_images = len(window_db)
    for im_i in xrange(num_images):
        targets = compute_bbox_regression_targets(window_db[im_i]['windows'])
        window_db[im_i]['bbox_targets'] = targets

    # Compute values needed for means and stds
    # var(x) = E(x^2) - E(x)^2
    class_counts = np.zeros((conf.NUM_CLASSES, 1)) + conf.EPS
    sums = np.zeros((conf.NUM_CLASSES, 4))
    squared_sums = np.zeros((conf.NUM_CLASSES, 4))
    maxes = -np.inf * np.ones((conf.NUM_CLASSES, 4))
    mins = np.inf * np.ones((conf.NUM_CLASSES, 4))
    for im_i in xrange(num_images):
        targets = window_db[im_i]['bbox_targets']
        for cls in xrange(1, conf.NUM_CLASSES):
            cls_inds = np.where(targets[:, 0] == cls)[0]
            if cls_inds.size > 0:
                class_counts[cls] += cls_inds.size
                sums[cls, :] += targets[cls_inds, 1:].sum(axis=0)
                squared_sums[cls, :] += (targets[cls_inds, 1:] ** 2).sum(axis=0)
                maxes[cls, :] = np.maximum(maxes[cls, :],
                                           targets[cls_inds, 1:].max(axis=0))
                mins[cls, :] = np.minimum(mins[cls, :],
                                          targets[cls_inds, 1:].min(axis=0))
    means = sums / class_counts
    stds = np.sqrt(squared_sums / class_counts - means ** 2)

    # Normalize targets
    for im_i in xrange(num_images):
        targets = window_db[im_i]['bbox_targets']
        for cls in xrange(1, conf.NUM_CLASSES):
            cls_inds = np.where(targets[:, 0] == cls)[0]
            window_db[im_i]['bbox_targets'][cls_inds, 1:] \
                    -= means[cls, :]
            window_db[im_i]['bbox_targets'][cls_inds, 1:] \
                    /= stds[cls, :]

    # TODO(rbg) remove this when everything is in python
    import scipy.io
    scipy.io.savemat('../rcnn/data/voc_2007_means_stds.mat',
                     {'means': means, 'stds': stds})
    # These values will be needed for making predictions
    # (the predicts will need to be unnormalized and uncentered)
    return means, stds
