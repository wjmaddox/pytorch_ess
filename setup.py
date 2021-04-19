from setuptools import setup
import os
import sys

_here = os.path.abspath(os.path.dirname(__file__))

if sys.version_info[0] < 3:
    with open(os.path.join(_here, 'README.rst')) as f:
        long_description = f.read()
else:
    with open(os.path.join(_here, 'README.rst'), encoding='utf-8') as f:
        long_description = f.read()

setup(
    name='pytorch_ess',
    version='alpha',
    description=('PyTorch elliptical slice sampling'),
    long_description=long_description,
    author='Wesley Maddox',
    author_email='wm326@cornell.edu',
    url='https://github.com/wjmaddox/pytorch_ess',
    license='MPL-2.0',
    packages=['pytorch_ess'],
   install_requires=[
    'torch>=1.6.0',
   ],
    include_package_data=True,
    classifiers=[
        'Development Status :: 0',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.7'],
)