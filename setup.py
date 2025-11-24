# setup.py
import os
from pathlib import Path
from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np   # now safe: numpy is guaranteed to be available

BASE_DIR = Path(__file__).parent.resolve()

CYTHON_BUILD_DIR = BASE_DIR / "cython_build"
COMPILED_DIR = BASE_DIR / "compiled"

CYTHON_BUILD_DIR.mkdir(exist_ok=True)
COMPILED_DIR.mkdir(exist_ok=True)

extensions = [
    Extension("data_structures", ["data_structures.pyx"], include_dirs=[np.get_include()]),
    Extension("bezier",          ["bezier.pyx"],          include_dirs=[np.get_include()]),
    Extension("rotation",        ["rotation.pyx"],        include_dirs=[np.get_include()]),
    Extension("triangulation",   ["triangulation.pyx"],   include_dirs=[np.get_include()]),
    Extension("rasterize",       ["rasterize.pyx"],       include_dirs=[np.get_include()]),
]

setup(
    ext_modules=cythonize(
        extensions,
        build_dir=str(CYTHON_BUILD_DIR),
        language_level=3,
    ),
    options={
        "build_ext": {
            "build_lib": str(COMPILED_DIR),
            "inplace": False,
        }
    },
)