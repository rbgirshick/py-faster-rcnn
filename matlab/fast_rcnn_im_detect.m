% --------------------------------------------------------
% Fast R-CNN
% Copyright (c) 2015 Microsoft
% Licensed under The MIT License [see LICENSE for details]
% Written by Ross Girshick
% --------------------------------------------------------

function dets = fast_rcnn_im_detect(model, im, boxes)
% Perform detection a Fast R-CNN network given an image and
% object proposals.

if model.init_key ~= caffe('get_init_key')
  error('You probably need call fast_rcnn_load_net() first.');
end

[im_batch, scales] = image_pyramid(im, model.pixel_means, false);

[feat_pyra_boxes, feat_pyra_levels] = project_im_rois(boxes, scales);
rois = cat(2, feat_pyra_levels, feat_pyra_boxes);
% Adjust to 0-based indexing and make roi info the fastest dimension
rois = rois - 1;
rois = permute(rois, [2 1]);

input_blobs = cell(2, 1);
input_blobs{1} = im_batch;
input_blobs{2} = rois;
th = tic();
blobs_out = caffe('forward', input_blobs);
fprintf('fwd: %.3fs\n', toc(th));

bbox_deltas = squeeze(blobs_out{1})';
probs = squeeze(blobs_out{2})';

num_classes = size(probs, 2);
dets = cell(num_classes - 1, 1);
NMS_THRESH = 0.3;
% class index 1 is __background__, so we don't return it
for j = 2:num_classes
  cls_probs = probs(:, j);
  cls_deltas = bbox_deltas(:, (1 + (j - 1) * 4):(j * 4));
  pred_boxes = bbox_pred(boxes, cls_deltas);
  cls_dets = [pred_boxes cls_probs];
  keep = nms(cls_dets, NMS_THRESH);
  cls_dets = cls_dets(keep, :);
  dets{j - 1} = cls_dets;
end

% ------------------------------------------------------------------------
function [batch, scales] = image_pyramid(im, pixel_means, multiscale)
% ------------------------------------------------------------------------
% Construct an image pyramid that's ready for feeding directly into caffe
if ~multiscale
  SCALES = [600];
  MAX_SIZE = 1000;
else
  SCALES = [1200 864 688 576 480];
  MAX_SIZE = 2000;
end
num_levels = length(SCALES);

im = single(im);
% Convert to BGR
im = im(:, :, [3 2 1]);
% Subtract mean (mean of the image mean--one mean per channel)
im = bsxfun(@minus, im, pixel_means);

im_orig = im;
im_size = min([size(im_orig, 1) size(im_orig, 2)]);
im_size_big = max([size(im_orig, 1) size(im_orig, 2)]);
scale_factors = SCALES ./ im_size;

max_size = [0 0 0];
for i = 1:num_levels
  if round(im_size_big * scale_factors(i)) > MAX_SIZE
    scale_factors(i) = MAX_SIZE / im_size_big;
  end
  ims{i} = imresize(im_orig, scale_factors(i), 'bilinear', ...
                    'antialiasing', false);
  max_size = max(cat(1, max_size, size(ims{i})), [], 1);
end

batch = zeros(max_size(2), max_size(1), 3, num_levels, 'single');
for i = 1:num_levels
  im = ims{i};
  im_sz = size(im);
  im_sz = im_sz(1:2);
  % Make width the fastest dimension (for caffe)
  im = permute(im, [2 1 3]);
  batch(1:im_sz(2), 1:im_sz(1), :, i) = im;
end
scales = scale_factors';

% ------------------------------------------------------------------------
function [boxes, levels] = project_im_rois(boxes, scales)
% ------------------------------------------------------------------------
widths = boxes(:,3) - boxes(:,1) + 1;
heights = boxes(:,4) - boxes(:,2) + 1;

areas = widths .* heights;
scaled_areas = bsxfun(@times, areas, (scales.^2)');
diff_areas = abs(scaled_areas - (224 * 224));
[~, levels] = min(diff_areas, [], 2);

boxes = boxes - 1;
boxes = bsxfun(@times, boxes, scales(levels));
boxes = boxes + 1;

% ------------------------------------------------------------------------
function pred_boxes = bbox_pred(boxes, bbox_deltas)
% ------------------------------------------------------------------------
if isempty(boxes)
  pred_boxes = [];
  return;
end

Y = bbox_deltas;

% Read out predictions
dst_ctr_x = Y(:, 1);
dst_ctr_y = Y(:, 2);
dst_scl_x = Y(:, 3);
dst_scl_y = Y(:, 4);

src_w = boxes(:, 3) - boxes(:, 1) + eps;
src_h = boxes(:, 4) - boxes(:, 2) + eps;
src_ctr_x = boxes(:, 1) + 0.5 * src_w;
src_ctr_y = boxes(:, 2) + 0.5 * src_h;

pred_ctr_x = (dst_ctr_x .* src_w) + src_ctr_x;
pred_ctr_y = (dst_ctr_y .* src_h) + src_ctr_y;
pred_w = exp(dst_scl_x) .* src_w;
pred_h = exp(dst_scl_y) .* src_h;
pred_boxes = [pred_ctr_x - 0.5 * pred_w, pred_ctr_y - 0.5 * pred_h, ...
              pred_ctr_x + 0.5 * pred_w, pred_ctr_y + 0.5 * pred_h];
