import numpy as np
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

cmdclass = {}
ext_modules = [
    Extension(
        "utils.cython_bbox",
        ["utils/bbox.pyx"]
    ),
#    Extension(
#        "saint_tony.cython_nms",
#        ["saint_tony/nms.pyx"]
#    )
]
cmdclass.update({'build_ext': build_ext})

setup(
    name='saint_tony',
    cmdclass=cmdclass,
    ext_modules=ext_modules,
    include_dirs=[np.get_include()]
)
