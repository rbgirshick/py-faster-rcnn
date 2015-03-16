#!/usr/bin/env python
import fast_rcnn_config as conf
import fast_rcnn_train
import caffe
import argparse
import numpy as np
import datasets.pascal_voc
import time
import sys

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
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    default_solver = './models/vgg16_solver.prototxt'

    if args.solver is None:
        args.solver = default_solver

    if args.pretrained_model is None:
        print('Warning: starting from random initialization')
        time.sleep(2)

    # fix the random seed for reproducibility
    np.random.seed(conf.RNG_SEED)

    # set up caffe
    caffe.set_mode_gpu()
    if args.gpu_id is not None:
        caffe.set_device(args.gpu_id)

    imdb_train = datasets.pascal_voc('trainval', '2007')
    print 'Loaded dataset `{:s}` for training'.format(imdb_train.name)

    fast_rcnn_train.train_net(args.solver, imdb_train,
                              pretrained_model=args.pretrained_model,
                              epochs=args.epochs)
