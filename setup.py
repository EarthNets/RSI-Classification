import os
import os.path as osp
import shutil
import sys
import warnings
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
        name = "RSI-Classification",
        version = "0.0.1",
        author = "Zhitong Xiong",
        author_email = "xiongzhitong@gmail.com",
        description = ("Dataset loaders for Remote Sensing datasts on Earth observation tasks."),
        license = "BSD",
        keywords = "Remote sensing image segmentation",
        url = "http://packages.python.org/RSI-Classification",
        packages=['rsicls', 'tests'],
        long_description=read('README.md'),
        classifiers=[
                    "Development Status :: 3 - Alpha",
                    "Topic :: Utilities",
                    "License :: OSI Approved :: BSD License",
                ],
)
