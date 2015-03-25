# Fast R-CNN

### Requirements

0. Requirements for Caffe and pycaffe
0. cython and python-opencv (`sudo apt-get install cython python-opencv` on Ubuntu)
0. Matlab (required for PASCAL VOC evaluation only)

### Installation

0. Extract the source code: `$ tar zxvf fast-rcnn.tgz`; call the source directory `$FRCNN`
0. Build Cython modules: `$ cd $FRCNN && make` (there will probably be some benign warnings that you can ignore)
0. Build Caffe: `$ cd $FRCNN/caffe-master` and follow the [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html)
0. Establish symlinks for VOCdevkits
  0. Symlink `$FRCNN/datasets/VOCdevkit2007` to where you have the PASCAL VOC 2007 devkit and data installed
  0. And similiarly for other PASCAL VOC 20xy datasets
  0. Symlink `$FRCNN/datasets/cache` to somewhere that will store cached dataset files
 
### Usage

Train a Fast R-CNN detector. For example, train a VGG 16 network on VOC 2007 trainval:

`./experiments/train_net.py --gpu 0 --solver models/VGG_16/solver.prototxt --weights /data/reference_caffe_nets/VGG_ILSVRC_16_layers.v2.caffemodel`

Test a Fast R-CNN detector. For example, test the VGG 16 network on VOC 2007 test:

`./experiments/test_net.py --gpu 1 --def models/VGG_16/test.prototxt --net snapshots/vgg16_fast_rcnn_iter_40000.caffemodel`

Test output is written underneath `$FRCNN/output`.
