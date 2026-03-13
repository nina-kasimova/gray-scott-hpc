from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("gray_scott_cython.pyx",annotate=True,
                          compiler_directives={"language_level": "3"}),
    include_dirs=[numpy.get_include()]
)
