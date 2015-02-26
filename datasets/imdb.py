#imdb.name = 'voc_train_2007'
#imdb.image_dir = '/work4/rbg/VOC2007/VOCdevkit/VOC2007/JPEGImages/'
#imdb.extension = '.jpg'
#imdb.image_ids = {'000001', ... }
#imdb.sizes = [numimages x 2]
#imdb.classes = {'aeroplane', ... }
#imdb.num_classes
#imdb.class_to_id
#imdb.class_ids
#imdb.eval_func = pointer to the function that evaluates detections
#imdb.roidb_func = pointer to the function that returns regions of interest

# imdb.name
# imdb.image_index
# imdb.image_path_from_index
# imdb.classes

import os

class imdb(object):
    def __init__(self, name):
        self._name = name
        self._num_classes = 0
        self._classes = []
        self._image_index = []
        self._obj_proposer = 'selective_search'
        self._roidb = None
        self._roidb_handler = self.default_roidb

    @property
    def name(self):
        return self._name

    @property
    def num_classes(self):
        return len(self._classes)

    @property
    def classes(self):
        return self._classes

    @property
    def image_index(self):
        return self._image_index

    @property
    def roidb_handler(self):
        return self._roidb_handler

    @roidb_handler.setter
    def roidb_handler(self, val):
        self._roidb_handler = val

    def image_path_at(self, i):
        raise NotImplementedError

    @property
    def roidb(self):
        if self._roidb is not None:
            return self._roidb
        self._roidb = self.roidb_handler()
        return self._roidb

    def default_roidb(self):
        raise NotImplementedError

    @property
    def cache_path(self):
        return os.path.abspath(os.path.join(
            os.path.dirname(__file__),
            'cache'))
