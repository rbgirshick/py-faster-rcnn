function cnn = init_cnn_model(varargin)
% cnn = init_cnn_model
% Initialize a CNN with caffe
%
% Optional arguments
% net_file   network binary file
% def_file   network prototxt file
% use_gpu    set to false to use CPU (default: true)
% use_caffe  set to false to avoid using caffe (default: true)
%            useful for running on the cluster (must use cached pyramids!)

% ------------------------------------------------------------------------
% Options
ip = inputParser;

% network binary file
ip.addParamValue('net_file', ...
    './data/caffe_nets/ilsvrc_2012_train_iter_310k', ...
    @isstr);

% network prototxt file
ip.addParamValue('def_file', ...
    './model-defs/pyramid_cnn_output_conv5_scales_7_plane_1713.prototxt', ...
    @isstr);

% Set use_gpu to false to use the CPU
ip.addParamValue('use_gpu', true, @islogical);

% Set use_caffe to false to avoid using caffe
% (must be used in conjunction with cached features!)
ip.addParamValue('use_caffe', true, @islogical);

ip.parse(varargin{:});
opts = ip.Results;
% ------------------------------------------------------------------------

cnn.binary_file = opts.net_file;
cnn.definition_file = opts.def_file;
cnn.init_key = -1;

% load the ilsvrc image mean
data_mean_file = 'ilsvrc_2012_mean.mat';
assert(exist(data_mean_file, 'file') ~= 0);
% input size business isn't likley necessary, but we're doing it
% to be consistent with previous experiments
ld = load(data_mean_file);
mu = ld.image_mean; clear ld;
input_size = 227;
off = floor((size(mu,1) - input_size)/2)+1;
%mu = mu(off:off+input_size-1, off:off+input_size-1, :);
%mu = sum(sum(mu, 1), 2) / size(mu, 1) / size(mu, 2);
cnn.mu = reshape([102.9801, 115.9465, 122.7717], [1 1 3]);

if opts.use_caffe
  cnn.init_key = ...
      caffe('init', cnn.definition_file, cnn.binary_file);
  caffe('set_phase_test');
  if opts.use_gpu
    caffe('set_mode_gpu');
  else
    caffe('set_mode_cpu');
  end
end
