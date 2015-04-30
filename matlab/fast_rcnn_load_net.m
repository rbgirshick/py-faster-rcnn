% --------------------------------------------------------
% Fast R-CNN
% Copyright (c) 2015 Microsoft
% Licensed under The MIT License [see LICENSE for details]
% Written by Ross Girshick
% --------------------------------------------------------

function model = fast_rcnn_load_net(def, net, use_gpu)
% Load a Fast R-CNN network.

init_key = caffe('init', def, net, 'test');
if exist('use_gpu', 'var') && ~use_gpu
  caffe('set_mode_cpu');
else
  caffe('set_mode_gpu');
end

model.init_key = init_key;
% model.stride is correct for the included models, but may not be correct
% for other models!
model.stride = 16;
model.pixel_means = reshape([102.9801, 115.9465, 122.7717], [1 1 3]);
