"""
Fast Cython code for NMS of detection bounding boxes.

Originally written by Kai Wang for the plex project:
https://github.com/shiaokai/plex

See nms.pyx.license.txt.
"""
import numpy as np
cimport numpy as np
cimport cython

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

cdef inline int int_max(int a, int b): return a if a >= b else b
cdef inline int int_min(int a, int b): return a if a <= b else b
cdef inline float float_max(float a, float b): return a if a >= b else b
cdef inline float float_min(float a, float b): return a if a <= b else b


def nms(np.ndarray[DTYPE_t, ndim=2] bbs, overlap_thr = 0.5):
    """
    NMS detection boxes.

    Parameters
    ----------
    bbs: (N, 5) ndarray of float32
        [xmin, ymin, xmax, ymax, score]
    overlap_thr: float32

    Returns
    -------
    keep_idxs: list of int
        Indices to keep in the original bbs.
    """
    if bbs.shape[0] == 0:
       return np.zeros((0,5), dtype=DTYPE)

    # sort bbs by score
    cdef np.ndarray[np.int_t, ndim=1] sidx = np.argsort(bbs[:,4])
    sidx = sidx[::-1]
    bbs = bbs[sidx,:]

    keep = [True] * bbs.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] bbs_start_x = bbs[:,0]
    cdef np.ndarray[DTYPE_t, ndim=1] bbs_start_y = bbs[:,1]
    cdef np.ndarray[DTYPE_t, ndim=1] bbs_end_x = bbs[:,2]
    cdef np.ndarray[DTYPE_t, ndim=1] bbs_end_y = bbs[:,3]
    cdef np.ndarray[DTYPE_t, ndim=1] bbs_areas = \
        (bbs_end_y - bbs_start_y + 1) * (bbs_end_x - bbs_start_x + 1)
    cdef int i, jj
    cdef DTYPE_t intersect_width, intersect_height
    cdef DTYPE_t intersect_area, union_area
    cdef DTYPE_t overlap
    cdef DTYPE_t bbs_end_x_i, bbs_end_y_i, bbs_areas_i
    cdef DTYPE_t bbs_start_x_i, bbs_start_y_i

    # start at highest scoring bb
    for i in range(bbs.shape[0]):
        if not(keep[i]):
            continue
        bbs_end_x_i = bbs_end_x[i]
        bbs_end_y_i = bbs_end_y[i]
        bbs_areas_i = bbs_areas[i]
        bbs_start_x_i = bbs_start_x[i]
        bbs_start_y_i = bbs_start_y[i]
        for jj in range(i+1, bbs.shape[0]):
            if not(keep[jj]):
                continue
            # mask out all worst scoring overlapping
            intersect_width = float_min(bbs_end_x_i, bbs_end_x[jj]) - \
                float_max(bbs_start_x_i, bbs_start_x[jj]) + 1
            if intersect_width <= 0:
                continue
            intersect_height = float_min(bbs_end_y_i, bbs_end_y[jj]) - \
                float_max(bbs_start_y_i, bbs_start_y[jj]) + 1
            if intersect_width <= 0:
                continue
            intersect_area = intersect_width * intersect_height
            union_area = bbs_areas_i + bbs_areas[jj] - intersect_area
            overlap = intersect_area / union_area
            # threshold and reject
            if overlap > overlap_thr:
                keep[jj] = False

    # Return original detection indices
    keep_idxs=[]
    for i in range(len(keep)):
        if keep[i]:
            keep_idxs.append(sidx[i])
    return keep_idxs
