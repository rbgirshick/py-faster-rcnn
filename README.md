# Fast R-CNN

### Requirements

1. Requirements for Caffe and pycaffe
2. Additional Python packages: cython, python-opencv, easydict
3. Matlab (required for PASCAL VOC evaluation only)

### Installation

1. Extract the source code: `$ tar zxvf fast-rcnn.tgz`; call the source directory $FRCNN
2. Build Cython modules: `$ cd $FRCNN/src && make` (there will probably be some benign warnings that you can ignore)
3. Build Caffe and pycaffe: `$ cd $FRCNN/caffe-fast-rcnn` and follow the Caffe installation instructions
4. Establish symlinks for VOCdevkits
  1. Symlink `$FRCNN/data/VOCdevkit2007` to where you have the PASCAL VOC 2007 devkit and data installed
  2. And similiarly for other PASCAL VOC 20XY datasets
  3. Symlink `$FRCNN/data/cache` to somewhere that will store cached dataset files

### Usage

Train a Fast R-CNN detector. For example, train a VGG 16 network on VOC 2007 trainval:

```
./tools/train_net.py --gpu 0 --solver models/VGG16/solver.prototxt --weights data/imagenet_models/VGG16.v2.caffemodel
```

Test a Fast R-CNN detector. For example, test the VGG 16 network on VOC 2007 test:

```
./tools/test_net.py --gpu 1 --def models/VGG16/test.prototxt --net output/voc_2007_trainval/vgg16_fast_rcnn_iter_40000.caffemodel
```

Test output is written underneath `$FRCNN/output`.

Compress a Fast R-CNN model using SVD on the fully-connected layers:

```
./tools/compress_model.py --def models/VGG16/test.prototxt --def-svd models/VGG16/compressed/test.prototxt --net output/voc_2007_trainval/vgg16_fast_rcnn_iter_40000.caffemodel
```
