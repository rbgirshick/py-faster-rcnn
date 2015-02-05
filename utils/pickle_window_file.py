#! /usr/bin/env python
import sys
import cPickle
import numpy as np
import gzip

def window_file_to_pickle(filename):
    window_db = []
    with open(filename) as f:
        count = 0
        while True:
            image_header = f.readline()
            if not image_header:
                break
            count += 1
            if count % 100 == 0:
                print 'reading line: %d' % (count + 1)
            image_path = f.readline().strip()
            channels = int(f.readline().strip())
            height = int(f.readline().strip())
            width = int(f.readline().strip())
            num_windows = int(f.readline().strip())
            windows = np.zeros((0, 6))
            for i in xrange(num_windows):
                window = [float(x) for x in f.readline().strip().split(' ')]
                windows = np.append(windows, [window], axis=0)
            window_db.append({'image': image_path,
                              'windows': windows,
                              'height': height,
                              'width': width})
    out_filename = filename + '.pz'
    print 'saving to file %s' % out_filename
    with gzip.GzipFile(out_filename, 'wb') as f:
        cPickle.dump(window_db, f, protocol=cPickle.HIGHEST_PROTOCOL)
    #f = open(out_filename, 'wb')
    #cPickle.dump(window_db, f, protocol=cPickle.HIGHEST_PROTOCOL)
    #f.close()

if __name__ == '__main__':
    window_file_to_pickle(sys.argv[1])
