import numpy as np
import cv2
import time

class DeepPyramid(object):
    def __init__(self, net=None):
        self.net = net
        self.stride = 16. # TODO(rbg): remove magic (conv5 stride)
        self.num_levels = 7
        self.scale_factor = 1.0 / np.sqrt(2.0)
        self.scales = np.array(self.scale_factor ** xrange(self.num_levels))
        self.base_size = 1713
        # FIXME: get these values from the net
        self.pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]])

    def get_image_pyramid(self, im):
        # NB 1: Assumes that im is in BGR order already!
        # NB 2: Subtracts the pixel mean
        im_pyra = []
        im_shape = im.shape
        im = im.astype(np.float32, copy=False)
        im -= self.pixel_means
        if im_shape[0] > im_shape[1]:
            base_scale = float(self.base_size) / im_shape[0]
            base_shape = (self.base_size,
                          int(np.round(im_shape[1] * base_scale)))
        else:
            base_scale = float(self.base_size) / im_shape[1]
            base_shape = (int(np.round(im_shape[0] * base_scale)),
                          self.base_size)
        # Ensure that the longest axis of the first level is always base_size
        # OpenCV assumes (width, height) instead of the usual (height, width)
        im_pyra.append(cv2.resize(im, base_shape[::-1]))
        for i in xrange(1, self.num_levels):
            scale_factor = base_scale * self.scales[i]
            im_pyra.append(cv2.resize(im, None, None,
                                      scale_factor, scale_factor))
        scales = base_scale * self.scales
        return im_pyra, scales

    def image_pyramid_to_batch(self, im_pyra):
        base_shape = im_pyra[0].shape
        caffe_input = np.zeros((self.num_levels, base_shape[0],
                                base_shape[1], base_shape[2]))
        for i in xrange(self.num_levels):
            im_ = im_pyra[i].astype(np.float32)
            caffe_input[i, 0:im_.shape[0], 0:im_.shape[1], :] = im_
        # Move channels (axis 3) to axis 1
        channel_swap = (0, 3, 1, 2)
        caffe_input = caffe_input.transpose(channel_swap)
        return caffe_input

    def get_feat_pyramid(self, im):
        im_pyra = self.get_image_pyramid(im)
        feat_pyra = []
        for i in xrange(self.num_levels):
            caffe_input = im_pyra[i].astype(np.float32)
            caffe_input = caffe_input.transpose((2, 0, 1))
            caffe_input = caffe_input[np.newaxis]
            self.net.blobs['data'].reshape(caffe_input.shape[0],
                                           caffe_input.shape[1],
                                           caffe_input.shape[2],
                                           caffe_input.shape[3])
            print caffe_input.shape
            start_t = time.time()
            blob = self.net.forward(data=caffe_input)
            feat_pyra.append(blob['conv5'].copy())
            print 'Elapsed time: {:.4f}'.format(time.time() - start_t)
        return feat_pyra

    def get_feat_pyramid_batch(self, im):
        im_pyra = self.get_image_pyramid(im)
        caffe_input = self.image_pyramid_to_batch(im_pyra)
        base_shape = im_pyra[0].shape
        self.net.blobs['data'].reshape(self.num_levels, base_shape[2],
                                       base_shape[0], base_shape[1])
        start_t = time.time()
        blob = self.net.forward(data=caffe_input)
        print 'Elapsed time: {:.4f}'.format(time.time() - start_t)
        return blob['conv5'].copy()

    def im_to_feat_pyramid_coords(self, im_rois, pyra_scales):
        widths = im_rois[:, 2] - im_rois[:, 0] + 1
        heights = im_rois[:, 3] - im_rois[:, 1] + 1

        sizes = np.maximum(widths, heights)

        # optimal scales
        opt_scales = 224. / sizes

        # compute D[i,:] = log(opt_scales) - log(scale at level i)
        # positive --> optimal scale is larger than pyramid scale
        scale_diff = np.ones((len(pyra_scales), 1)) * \
                     np.log(opt_scales[np.newaxis, :]) - \
                     np.log(pyra_scales[:, np.newaxis]) * \
                     np.ones((1, len(opt_scales)))

        # each column is a ROI
        # each row is a scale
        scale_diff = np.where(scale_diff < 0, np.inf, scale_diff)
        pyra_levels = scale_diff.argmin(axis=0)

        # make sure that we always pick a scale for which
        # the max length of the proposal will be <= 224 (spans 15 feature cells)
        #assert(all(opt_scales >= scales(levels)));

        scale_xform = pyra_scales[pyra_levels]
        pyra_rois = im_rois * scale_xform[:, np.newaxis]
        pyra_rois /= self.stride
        pyra_rois = np.round(pyra_rois)

        #w = boxes(:, 3) - boxes(:, 1) + 1;
        #h = boxes(:, 4) - boxes(:, 2) + 1;
        #assert(all(w <= 15));
        #assert(all(h <= 15));
        return pyra_levels, pyra_rois

#    def score_boxes(self, im, boxes):
        # Convert boxes to pyramid rois formatted as num_rois x roi_desc
        # where roi_desc is [level x1 y1 x2 y2]
        # Construct image pyramid and pass image pyramid input blob and roi blob
        # to caffe for forward pass.
