from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("mincall/input_readers.pyx")
)
