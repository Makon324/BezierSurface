from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np
import os

# ------------------------------------------------------------------
# 1. Where to put all generated .c files and build temp files
# ------------------------------------------------------------------
CYTHON_BUILD_DIR = "cython_build"  # .c files go here
os.makedirs(CYTHON_BUILD_DIR, exist_ok=True)

# ------------------------------------------------------------------
# 2. Where to put the final compiled extensions (.so / .pyd)
# ------------------------------------------------------------------
COMPILED_DIR = "compiled"  # <-- this is your new subdir
os.makedirs(COMPILED_DIR, exist_ok=True)

extensions = [
    Extension(
        "data_structures",
        ["data_structures.pyx"],
        include_dirs=[np.get_include()],
    ),
    Extension("bezier", ["bezier.pyx"], include_dirs=[np.get_include()]),
    Extension("rotation", ["rotation.pyx"], include_dirs=[np.get_include()]),
    Extension("triangulation", ["triangulation.pyx"], include_dirs=[np.get_include()]),
    Extension("rasterize", ["rasterize.pyx"], include_dirs=[np.get_include()]),
]

setup(
    ext_modules=cythonize(
        extensions,
        build_dir=CYTHON_BUILD_DIR,  # keeps all .c files out of source tree
        language_level="3",
    ),
    options={
        "build_ext": {
            "inplace": False,  # CRUCIAL: don't drop .so in current dir
            "build_lib": COMPILED_DIR,  # put final extensions here
        }
    },
)
