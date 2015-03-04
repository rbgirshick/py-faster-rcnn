#!/usr/bin/env python

import sys
import subprocess
caffe_path = '../caffe/python'
sys.path.insert(0, caffe_path)

import argparse
from utils.timer import Timer
import numpy as np
import matplotlib.pyplot as plt
import cv2
import caffe
import fast_rcnn_config as conf
import utils.cython_nms
import datasets.pascal_voc
import cPickle
import heapq

def _get_image_blob(im):
    im_pyra = []
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= conf.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    max_shape = (0, 0, 0)
    processed_ims = []
    im_scale_factors = []

    for target_size in conf.TEST_SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > conf.TEST_MAX_SIZE:
            im_scale = float(conf.TEST_MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)
        max_shape = np.maximum(max_shape, im.shape)

    num_images = len(processed_ims)
    blob = np.zeros((num_images, max_shape[0],
                     max_shape[1], max_shape[2]), dtype=np.float32)
    for i in xrange(num_images):
        im = processed_ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
    # Move channels (axis 3) to axis 1
    # Axis order will become: (batch elem, channel, height, width)
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)
    return blob, np.array(im_scale_factors)

def _get_rois_blob(im_rois, im_scale_factors):
    feat_rois, levels = _map_im_rois_to_feat_rois(im_rois, im_scale_factors)
    rois_blob = np.hstack((levels, feat_rois))[:, :, np.newaxis, np.newaxis]
    return rois_blob.astype(np.float32, copy=False)

def _map_im_rois_to_feat_rois(im_rois, scales):
    im_rois = im_rois.astype(np.float, copy=False)
    widths = im_rois[:, 2] - im_rois[:, 0] + 1
    heights = im_rois[:, 3] - im_rois[:, 1] + 1

    areas = widths * heights
    scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
    # TODO(rbg): 227 or 224? or blah/....
    diff_areas = np.abs(scaled_areas - 227 * 227)
    levels = diff_areas.argmin(axis=1)[:, np.newaxis]

    feat_rois = np.round(im_rois * scales[levels] / conf.FEAT_STRIDE)
    return feat_rois, levels

def _get_blobs(im, rois):
    blobs = {'data' : None, 'rois' : None}
    blobs['data'], im_scale_factors = _get_image_blob(im)
    blobs['rois'] = _get_rois_blob(rois, im_scale_factors)
    return blobs, im_scale_factors

def _bbox_pred(boxes, box_deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0, box_deltas.shape[1]))

    boxes = boxes.astype(np.float, copy=False)
    widths = boxes[:, 2] - boxes[:, 0] + conf.EPS
    heights = boxes[:, 3] - boxes[:, 1] + conf.EPS
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = box_deltas[:, 0::4]
    dy = box_deltas[:, 1::4]
    dw = box_deltas[:, 2::4]
    dh = box_deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(box_deltas.shape)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h
    return pred_boxes

def _clip_boxes(boxes, im_shape):
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
    return boxes

def im_detect(net, im, boxes):
    # TODO: remove duplicates
    blobs, im_scale_factors = _get_blobs(im, boxes)

    # v = np.array([1, 1e3, 1e6, 1e9, 1e12])
    # hashes = blobs['rois'][:, :, 0, 0].dot(v.T)
    hashes = (blobs['rois'][:, :, 0, 0] *
              np.array([[1, 1e3, 1e6, 1e9, 1e12]])).sum(axis=1)
    _, index, inv_index = np.unique(hashes, return_index=True,
                                    return_inverse=True)
    blobs['rois'] = blobs['rois'][index, :, :, :]
    boxes = boxes[index, :]

    # reshape network inputs
    base_shape = blobs['data'].shape
    num_rois = blobs['rois'].shape[0]
    net.blobs['data'].reshape(base_shape[0], base_shape[1],
                              base_shape[2], base_shape[3])
    net.blobs['rois'].reshape(num_rois, 5, 1, 1)
    blobs_out = net.forward(data=blobs['data'].astype(np.float32, copy=False),
                            rois=blobs['rois'].astype(np.float32, copy=False))
    scores = blobs_out['fc8_pascal'][:, :, 0, 0]
    # Return scores as fg - bg
    scores = scores - scores[:, 0][:, np.newaxis]
    box_deltas = blobs_out['fc8_pascal_bbox'][:, :, 0, 0]
    pred_boxes = _bbox_pred(boxes, box_deltas)
    pred_boxes = _clip_boxes(pred_boxes, im.shape)

    scores = scores[inv_index, :]
    pred_boxes = pred_boxes[inv_index, :]

    # TODO(rbg): try variant where we predict boxes and then score those
    # Need to compute all cls_rois and then deduplicate
#    for i in xrange(1, scores.shape[1]):
#        cls_rois_blob = _get_rois_blob(pred_boxes[:, i*4:(i+1)*4],
#                                       im_scale_factors)
#        t = Timer()
#        t.tic()
#        blobs_out = net.forward(data=blobs['data'].astype(np.float32,
#                                                          copy=False),
#                                rois=cls_rois_blob.astype(np.float32,
#                                                          copy=False),
#                                start='roi_pool5')
#        print t.toc()
#        cls_scores = blobs_out['fc8_pascal'][:, :, 0, 0]
#        scores[:, i] = cls_scores[:, i] - cls_scores[:, 0]

    return scores, pred_boxes

def _vis_detections(im, class_name, dets):
    im = im[:, :, (2, 1, 0)]
    for i in xrange(np.minimum(10, dets.shape[0])):
        bbox = dets[i, :4]
        score = dets[i, -1]
        if score > 0:
            plt.cla()
            plt.imshow(im)
            plt.gca().add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='g', linewidth=3)
                )
            plt.title('{}  {:.3f}'.format(class_name, score))
            plt.pause(0.5)

def _write_voc_results_file(imdb, all_boxes):
    pid = os.getpid()
    #/data/VOC2007/VOCdevkit/results/VOC2007/Main/comp4-44503_det_test_aeroplane.txt
    base_path = './datasets/VOCdevkit2007/results/VOC2007/Main/comp4-{}_'.format(pid)
    for cls_ind, cls in enumerate(imdb.classes):
        if cls == '__background__':
            continue
        file_name = base_path + 'det_test_' + cls + '.txt'
        with open(file_name, 'wt') as f:
            for im_ind, index in enumerate(imdb.image_index):
                dets = all_boxes[cls_ind][im_ind]
                if dets == []:
                    continue
                keep = utils.cython_nms.nms(dets, 0.3)
                if len(keep) == 0:
                    continue
                dets = dets[keep, :]
                # the VOCdevkit expects 1-based indices
                dets[:, :4] += 1
                for k in xrange(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(
                            index, dets[k, -1], dets[k, 0], dets[k, 1],
                            dets[k, 2], dets[k, 3]))
    print 'Evaluate comp4-{}'.format(pid)
    return pid

def _do_matlab_eval(pid):
    cmd = 'cd ../rcnn;'
    cmd += 'matlab -nodisplay -nodesktop '
    cmd += '-r "load imdb/cache/imdb_voc_2007_test.mat; '
    cmd += 'imdb_eval_voc_py(imdb, {});"'.format(pid)
    status = subprocess.call(cmd), shell=True)

def fast_rcnn_test(net, imdb):
    num_images = len(imdb.image_index)
    # heuristic: keep an average of 40 detections per class per images prior
    # to NMS
    max_per_set = 40 * num_images
    # heuristic: keep at most 100 detection per class per image prior to NMS
    max_per_image = 100
    # detection thresold for each class (this is adaptively set based on the
    # max_per_set constraint)
    thresh = -np.inf * np.ones(imdb.num_classes)
    # top_scores will hold one minheap of scores per class (used to enforce
    # the max_per_set constraint)
    top_scores = [[] for _ in xrange(imdb.num_classes)]
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}

    roidb = imdb.roidb
    for i in xrange(num_images):
        im = cv2.imread(imdb.image_path_at(i))
        _t['im_detect'].tic()
        scores, boxes = im_detect(net, im, roidb[i]['boxes'])
        _t['im_detect'].toc()

        _t['misc'].tic()
        for j in xrange(1, imdb.num_classes):
            inds = np.where((scores[:, j] > thresh[j]) &
                            (roidb[i]['gt_classes'] == 0))[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j*4:(j+1)*4]
            top_inds = np.argsort(-cls_scores)[:max_per_image]
            cls_scores = cls_scores[top_inds]
            cls_boxes = cls_boxes[top_inds, :]
            # push new scores onto the minheap
            for val in cls_scores:
                heapq.heappush(top_scores[j], val)
            # if we've collected more than the max number of detection,
            # then pop items off the minheap and update the class threshold
            if len(top_scores[j]) > max_per_set:
                while len(top_scores[j]) > max_per_set:
                    heapq.heappop(top_scores[j])
                thresh[j] = top_scores[j][0]

            all_boxes[j][i] = \
                    np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                    .astype(np.float32, copy=False)

            if 0:
                keep = utils.cython_nms.nms(all_boxes[j][i], 0.3)
                _vis_detections(im, imdb.classes[j], all_boxes[j][i][keep, :])
        _t['misc'].toc()

        print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_images, _t['im_detect'].average_time,
                      _t['misc'].average_time)

    for j in xrange(1, imdb.num_classes):
        for i in xrange(num_images):
            inds = np.where(all_boxes[j][i][:, -1] > thresh[j])[0]
            all_boxes[j][i] = all_boxes[j][i][inds, :]

    with open('dets.pkl', 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    pid = _write_voc_results_file(imdb, all_boxes)
    _do_matlab_eval(pid)

    # Write results file and call matlab to evaluate

if __name__ == '__main__':
    prototxt = 'model-defs/vgg16_pyramid_forward_only_bbox_reg.prototxt'
    caffemodel = '/home/rbg/working/pyramid-rcnn/fast-rcnn/snapshots/vgg16_finetune_all_joint_bbox_reg_smoothL1_roidb2_iter_40000.caffemodel'

    caffe.set_phase_test()
    caffe.set_mode_gpu()
    GPU_ID = 2
    if GPU_ID is not None:
        caffe.set_device(GPU_ID)
    net = caffe.Net(prototxt, caffemodel)

    import datasets.pascal_voc
    imdb = datasets.pascal_voc('test', '2007')
    fast_rcnn_test(net, imdb)
