#!/usr/bin/env python
import fast_rcnn_config as conf
import fast_rcnn_test
import caffe

if __name__ == '__main__':
    prototxt = 'model-defs/vgg16_bbox_reg_deploy.prototxt'
    caffemodel = 'snapshots/vgg16_finetune_joint_flipped_iter_40000.caffemodel'
    GPU_ID = 2

    caffe.set_phase_test()
    caffe.set_mode_gpu()
    caffe.set_device(GPU_ID)
    net = caffe.Net(prototxt, caffemodel)

    import datasets.pascal_voc
    imdb = datasets.pascal_voc('test', '2007')
    fast_rcnn_test.test_net(net, imdb)
