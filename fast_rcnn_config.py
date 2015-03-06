import os
import sys
import numpy as np
caffe_path = os.path.abspath(os.path.join('..', 'caffe', 'python'))
sys.path.insert(0, caffe_path)

# Scales used in the SPP-net paper
# SCALES          = (480, 576, 688, 864, 1200)
SCALES          = (480, 576, 600)

# Max pixel size of a scaled input image
# MAX_SIZE        = 2000
MAX_SIZE        = 1000

# Images per batch
IMS_PER_BATCH   = 2 # 4

# Minibatch size
BATCH_SIZE      = 128 # 128

# Fraction of minibatch that is foreground labeled (class > 0)
FG_FRACTION     = 0.25

# Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
FG_THRESH       = 0.5

# Overlap threshold for a ROI to be considered background (class = 0 if
# overlap in [0.1, 0.5))
BG_THRESH_HI    = 0.5
BG_THRESH_LO    = 0.1

# Pixel mean values (BGR order) as a (1, 1, 3) array
PIXEL_MEANS     = np.array([[[102.9801, 115.9465, 122.7717]]])

# Stride in input image pixels at ROI pooling level (network specific)
# 16 is true for AlexNet and VGG-16
FEAT_STRIDE     = 16
BBOX_THRESH     = 0.5
EPS             = 1e-14
SNAPSHOT_ITERS  = 10000

TEST_SCALES     = (600,)
TEST_MAX_SIZE   = 1000
TEST_NMS        = 0.3
TEST_BINARY     = False
