#!/usr/bin/env python
import fast_rcnn_config as conf
import fast_rcnn_train
import caffe
import argparse
import numpy as np
import datasets.pascal_voc

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a fast R-CNN')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--solver', dest='solver', help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--epochs', dest='epochs',
                        help='number of epoch to train',
                        default=16, type=int)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    pretrained_model = \
        '/data/reference_caffe_nets/VGG_ILSVRC_16_layers.caffemodel'
    solver_prototxt = './models/vgg16_solver.prototxt'

    # fix the random seed for reproducibility
    np.random.seed(conf.RNG_SEED)

    args = parse_args()

    # set up caffe
    caffe.set_phase_train()
    caffe.set_mode_gpu()
    if args.gpu_id is not None:
        caffe.set_device(args.gpu_id)

    imdb_train = datasets.pascal_voc('trainval', '2007')
    print 'Loaded dataset `{:s}` for training'.format(imdb_train.name)

    if args.solver is None:
        args.solver = solver_prototxt

    fast_rcnn_train.train_net(args.solver, imdb_train,
                              pretrained_model=pretrained_model,
                              epochs=args.epochs)
