from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy as np

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='biss',
    version='1.0.0',
    author='Martin Weigl, Adrian Slowak, Emma Kiemeyer, Ines Langthallner, Daniel Ritzberger',
    author_email='martin.weigl@fh-ooe.at',
    description='Segmentation of brain vessels in MRI-Images using NeuronalNet',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://gitlab.fh-ooe.at/bin/biss',
    packages=['biss/backend', 'biss/frontend', 'biss/backend/region_grow'],
    ext_modules=cythonize(["src/biss/backend/region_grow/*.pyx"]),
    include_dirs=[np.get_include()],
    license='MIT',
    package_dir = {'': 'src'},
    install_requires=['imagecodecs', 'numpy', 'scikit-image', 'focal_loss', 'tensorflow', 'keras', 'yaspin'],
    package_data={'biss/backend/region_grow': ['*.pyx', '*.pxd']},
    include_package_data=True
)
