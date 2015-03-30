__sets = {}

import datasets.pascal_voc
for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = lambda split=split, year=year: \
               datasets.pascal_voc(split, year)

def get_imdb(name):
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()
