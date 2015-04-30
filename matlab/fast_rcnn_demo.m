% --------------------------------------------------------
% Fast R-CNN
% Copyright (c) 2015 Microsoft
% Licensed under The MIT License [see LICENSE for details]
% Written by Ross Girshick
% --------------------------------------------------------

function fast_rcnn_demo()
% Fast R-CNN demo (in matlab).

[folder, name, ext] = fileparts(mfilename('fullpath'));

caffe_path = fullfile(folder, '..', 'caffe-fast-rcnn', 'matlab', 'caffe');
addpath(caffe_path);

use_gpu = true;
% You can try other models here:
def = fullfile(folder, '..', 'models', 'VGG16', 'test.prototxt');;
net = fullfile(folder, '..', 'data', 'fast_rcnn_models', ...
               'vgg16_fast_rcnn_iter_40000.caffemodel');
model = fast_rcnn_load_net(def, net, use_gpu);

car_ind = 7;
sofa_ind = 18;
tv_ind = 20;

demo(model, '000004', [car_ind], {'car'});
demo(model, '001551', [sofa_ind, tv_ind], {'sofa', 'tvmonitor'});
fprintf('\n');

% ------------------------------------------------------------------------
function demo(model, im_id, cls_inds, cls_names)
% ------------------------------------------------------------------------
[folder, name, ext] = fileparts(mfilename('fullpath'));
box_file = fullfile(folder, '..', 'data', 'demo', [im_id '_boxes.mat']);
% Boxes were saved with 0-based indexing
ld = load(box_file); boxes = single(ld.boxes) + 1; clear ld;
im_file = fullfile(folder, '..', 'data', 'demo', [im_id '.jpg']);
im = imread(im_file);
dets = fast_rcnn_im_detect(model, im, boxes);

THRESH = 0.8;
for j = 1:length(cls_inds)
  cls_ind = cls_inds(j);
  cls_name = cls_names{j};
  I = find(dets{cls_ind}(:, end) >= THRESH);
  showboxes(im, dets{cls_ind}(I, :));
  title(sprintf('%s detections with p(%s | box) >= %.3f', ...
                cls_name, cls_name, THRESH))
  fprintf('\n> Press any key to continue');
  pause;
end
