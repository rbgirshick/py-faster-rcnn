# *Fast* R-CNN

Created by Ross Girshick at Microsoft Research, Redmond.

### Introduction

**Fast R-CNN** is a fast framework for object detection with deep ConvNets. Fast R-CNN
 - trains state-of-the-art models, like VGG16, 9x faster than traditional R-CNN and 3x faster than SPPnet,
 - runs 200x faster than R-CNN and 10x faster than SPPnet at test-time,
 - has a significantly higher mAP on PASCAL VOC than both R-CNN and SPPnet,
 - and is written in Python and C++/Caffe.

Fast R-CNN was initially described in an [arXiv tech report](http://arxiv.org/abs/1504.08083).

### License

Fast R-CNN is released under the MIT License (refer to the LICENSE file for details).

### Citing Fast R-CNN

If you find Fast R-CNN useful in your research, please consider citing:

    @article{girshick15fastrcnn,
        Author = {Ross Girshick},
        Title = {Fast R-CNN},
        Journal = {arXiv preprint arXiv:1504.08083},
        Year = {2015}
    }
    
### Contents
1. [Requirements: software](#requirements-software)
2. [Requirements: hardware](#requirements-hardware)
3. [Basic installation](#installation-sufficient-for-the-demo)
4. [Demo](#demo)
5. [Beyond the demo: training and testing](#beyond-the-demo-installation-for-training-and-testing-models)
6. [Usage](#usage)
7. [Extra downloads](#extra-downloads)

### Requirements: software

1. Requirements for `Caffe` and `pycaffe` (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html))

  **Note:** Caffe *must* be built with support for Python layers!

  ```make
  # In your Makefile.config, make sure to have this line uncommented
  WITH_PYTHON_LAYER := 1
  ```

  You can download my [Makefile.config](http://www.cs.berkeley.edu/~rbg/fast-rcnn-data/Makefile.config) for reference.
2. Python packages you might not have: `cython`, `python-opencv`, `easydict`
3. [optional] MATLAB (required for PASCAL VOC evaluation only)

### Requirements: hardware

1. For training smaller networks (CaffeNet, VGG_CNN_M_1024) a good GPU (e.g., Titan, K20, K40, ...) with at least 3G of memory suffices
2. For training with VGG16, you'll need a K40 (~11G of memory)

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

*After successfully completing [basic installation](#installation-sufficient-for-the-demo)*, you'll be ready to run the demo.

**Python**

To run the demo
```Shell
cd $FRCN_ROOT
./tools/demo.py
```
The demo performs detection using a VGG16 network trained for detection on PASCAL VOC 2007. The object proposals are pre-computed in order to reduce installation requirements.

**Note:** If the demo crashes Caffe because your GPU doesn't have enough memory, try running the demo with a small network, e.g., `./tools/demo.py --net caffenet` or with `--net vgg_cnn_m_1024`. Or run in CPU mode `./tools/demo.py --cpu`. Type `./tools/demo.py -h` for usage.

**MATLAB**

There's also a *basic* MATLAB demo, though it's missing some minor bells and whistles compared to the Python version.
```Shell
cd $FRCN_ROOT/matlab
matlab # wait for matlab to start...

# At the matlab prompt, run the script:
>> fast_rcnn_demo
```

Fast R-CNN training is implemented in Python only, but test-time detection functionality also exists in MATLAB.
See `matlab/fast_rcnn_demo.m` and `matlab/fast_rcnn_im_detect.m` for details.

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
  	
4. Create symlinks for the PASCAL VOC dataset

	```Shell
    cd $FRCN_ROOT/data
    ln -s $VOCdevkit VOCdevkit2007
    ```
    Using symlinks is a good idea because you will likely want to share the same PASCAL dataset installation between multiple projects.
5. [Optional] follow similar steps to get PASCAL VOC 2010 and 2012
6. Follow the next sections to download pre-computed object proposals and pre-trained ImageNet models

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

If you see this error

```
EnvironmentError: MATLAB command 'matlab' not found. Please add 'matlab' to your PATH.
```

then you need to make sure the `matlab` binary is in your `$PATH`. MATLAB is currently required for PASCAL VOC evaluation.

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
Scripts to reproduce the experiments in the paper (*up to stochastic variation*) are provided in `$FRCN_ROOT/experiments/scripts`. Log files for experiments are located in `experiments/logs`.

### Extra downloads

- [Experiment logs](http://www.cs.berkeley.edu/~rbg/fast-rcnn-data/fast_rcnn_experiments.tgz)
- PASCAL VOC test set detections
    - [voc_2007_test_results_fast_rcnn_caffenet_trained_on_2007_trainval.tgz](http://www.cs.berkeley.edu/~rbg/fast-rcnn-data/voc_2007_test_results_fast_rcnn_caffenet_trained_on_2007_trainval.tgz)
    - [voc_2007_test_results_fast_rcnn_vgg16_trained_on_2007_trainval.tgz](http://www.cs.berkeley.edu/~rbg/fast-rcnn-data/voc_2007_test_results_fast_rcnn_vgg16_trained_on_2007_trainval.tgz)
    - [voc_2007_test_results_fast_rcnn_vgg_cnn_m_1024_trained_on_2007_trainval.tgz](http://www.cs.berkeley.edu/~rbg/fast-rcnn-data/voc_2007_test_results_fast_rcnn_vgg_cnn_m_1024_trained_on_2007_trainval.tgz)
    - [voc_2012_test_results_fast_rcnn_vgg16_trained_on_2007_trainvaltest_2012_trainval.tgz](http://www.cs.berkeley.edu/~rbg/fast-rcnn-data/voc_2012_test_results_fast_rcnn_vgg16_trained_on_2007_trainvaltest_2012_trainval.tgz)
    - [voc_2012_test_results_fast_rcnn_vgg16_trained_on_2012_trainval.tgz](http://www.cs.berkeley.edu/~rbg/fast-rcnn-data/voc_2012_test_results_fast_rcnn_vgg16_trained_on_2012_trainval.tgz)
- [Fast R-CNN VGG16 model](http://www.cs.berkeley.edu/~rbg/fast-rcnn-data/voc12_submission.tgz) trained on VOC07 train,val,test union with VOC12 train,val
