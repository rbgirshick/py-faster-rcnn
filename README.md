# *Fast* R-CNN

Created by Ross Girshick at Microsoft Research, Redmond.

### Introduction

*Fast R-CNN* is a clean and fast framework for object detection.
Compared to traditional R-CNN, and its accelerated version SPPnet, Fast R-CNN trains networks using a multi-task loss in a single fine-tuning run.
The multi-task loss simplifies and speeds up training.
Unlike SPPnet, all network layers can be learned during fine-tuning.
We show that this difference has practical ramifications for very deep networks,  such as VGG16, where mAP suffers when only the fully-connected layers are   fine-tuned.
Compared to "slow" R-CNN, Fast R-CNN is 9x faster at training VGG16 for detection, 213x faster for detection, and achieves a significantly higher mAP on PASCAL VOC 2012.
Compared to SPPnet, Fast R-CNN trains VGG16 3x faster, tests 10x faster, and is more accurate.

Fast R-CNN was initially described in an [arXiv tech report](http://arxiv.org/abs/todo).

### Citing Fast R-CNN

If you find R-CNN useful in your research, please consider citing:

    @article{girshick15fastrcnn,
        Author = {Ross Girshick},
        Title = {Fast R-CNN},
        Journal = {arXiv preprint arXiv:todo},
        Year = {2015}
    }

### License

Fast R-CNN is released under the MIT License (refer to the LICENSE file for details).

### Installation requirements

1. Requirements for Caffe and pycaffe (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html))
2. Additional Python packages: cython, python-opencv, easydict
3. [optional] MATLAB (required for PASCAL VOC evaluation only)

### Installation (sufficient for the demo)

1. Clone the Fast R-CNN repository
  ```Shell
  # Make sure to clone with --recursive
  git clone --recursive git@github.com:rbgirshick/fast-rcnn.git
  ```
  
2. We'll call the directory that you cloned Fast R-CNN into `FRCN_ROOT`
3. Build the Cython modules
    ```Shell
    cd $FRCN_ROOT/lib
    make
    ```
    
4. Build Caffe and pycaffe
    ```Shell
    cd $FRCN_ROOT/caffe-fast-rcnn
    # Now follow the Caffe installation instructions here:
    #   http://caffe.berkeleyvision.org/installation.html

    # If you're experienced with Caffe and have all of the requirements installed
    # and your Makefile.config in place, then simply do:
    make -j8 && make pycaffe
    ```
    
5. Download pre-computed Fast R-CNN detectors
    ```Shell
    cd $FRCN_ROOT
    ./data/scripts/fetch_fast_rcnn_models.sh
    ```

    This will populate the `$FRCN_ROOT/data` folder with `fast_rcnn_models`. See `data/README.md` for details.

### Demo

To run the demo
```Shell
cd $FRCN_ROOT
./tools/demo.py
```
The demo performs detection using a VGG16 network trained for detection on PASCAL VOC 2012. The object proposals are pre-computed in order to reduce installation requirements.

### Beyond the demo: installation for training and testing models
1. Download the training, validation, test data and VOCdevkit

	```Shell
	wget http://pascallin.ecs.soton.ac.uk/challenges/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
	wget http://pascallin.ecs.soton.ac.uk/challenges/VOC/voc2007/VOCtest_06-Nov-2007.tar
	wget http://pascallin.ecs.soton.ac.uk/challenges/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
	```
	
2. Extract all of these tars into one directory named `VOCdevkit`

	```Shell
	tar xvf VOCtrainval_06-Nov-2007.tar
	tar xvf VOCtest_06-Nov-2007.tar
	tar xvf VOCdevkit_08-Jun-2007.tar
	```

3. It should have this basic structure

	```Shell
  	$VOCdevkit/                           # development kit
  	$VOCdevkit/VOCcode/                   # VOC utility code
  	$VOCdevkit/VOC2007                    # image sets, annotations, etc.
  	# ... and several other directories ...
  	```
  	
4. Establish symlinks for the PASCAL VOC dataset

	```Shell
    cd $FRCN_ROOT/data
    ln -s $VOCdevkit VOCdevkit2007
    ```
    
5. Establish a symlink for your cache directory

	```Shell
    cd $FRCN_ROOT/data
    # /your/cache/path needs to be a directory that will hold a few GB of data
    ln -s /your/cache/path cache
    ```
    
6. [Optional] follow similar steps to get PASCAL VOC 2010 and 2012
7. Follow the next sections to download pre-computed object proposals and pre-trained ImageNet models

### Download pre-computed Selective Search object proposals

Pre-computed selective search boxes can also be downloaded for VOC2007 and VOC2012.

```Shell
cd $FRCN_ROOT
./data/scripts/fetch_selective_search_data.sh
```

This will populate the `$FRCN_ROOT/data` folder with `selective_selective_data`.

### Download pre-trained ImageNet models

Pre-trained ImageNet models can be downloaded for the three networks described in the paper: CaffeNet (model **S**), VGG_CNN_M_1024 (model **M**), and VGG16 (model **L**).

```Shell
cd $FRCN_ROOT
./data/scripts/fetch_imagenet_models.sh
```
These models are all available in the [Caffe Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo), but are provided here for your convenience.

### Usage

**Train** a Fast R-CNN detector. For example, train a VGG16 network on VOC 2007 trainval:

```Shell
./tools/train_net.py --gpu 0 --solver models/VGG16/solver.prototxt \
	--weights data/imagenet_models/VGG16.v2.caffemodel
```

**Test** a Fast R-CNN detector. For example, test the VGG 16 network on VOC 2007 test:

```Shell
./tools/test_net.py --gpu 1 --def models/VGG16/test.prototxt \
	--net output/default/voc_2007_trainval/vgg16_fast_rcnn_iter_40000.caffemodel
```

Test output is written underneath `$FRCN_ROOT/output`.

**Compress** a Fast R-CNN model using truncated SVD on the fully-connected layers:

```Shell
./tools/compress_net.py --def models/VGG16/test.prototxt \
	--def-svd models/VGG16/compressed/test.prototxt \
    --net output/default/voc_2007_trainval/vgg16_fast_rcnn_iter_40000.caffemodel
# Test the model you just compressed
./tools/test_net.py --gpu 0 --def models/VGG16/compressed/test.prototxt \
	--net output/default/voc_2007_trainval/vgg16_fast_rcnn_iter_40000_svd_fc6_1024_fc7_256.caffemodel
```

### Experiment scripts
Scripts to reproduce the experiments in the paper (up to stochastic variation) are provided in `$FRCN_ROOT/experiments/scripts`. Log files for experiments are located in `experiments/logs`.
