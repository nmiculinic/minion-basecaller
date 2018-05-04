from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("mincall/train/_input_feeders.pyx")
)
