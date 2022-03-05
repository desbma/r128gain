#!/usr/bin/env python3

""" Package setup. """

import os
import re
import sys

from setuptools import find_packages, setup

if sys.hexversion < 0x3060000:
    print("Python version %s is unsupported, >= 3.6.0 is needed" % (".".join(map(str, sys.version_info[:3]))))
    exit(1)

with open(os.path.join("r128gain", "__init__.py"), "rt") as f:
    version_match = re.search('__version__ = "([^"]+)"', f.read())
assert version_match is not None
version = version_match.group(1)

with open("requirements.txt", "rt") as f:
    requirements = f.read().splitlines()

with open("test-requirements.txt", "rt") as f:
    test_requirements = f.read().splitlines()

with open("README.md", "rt") as f:
    readme = f.read()

setup(
    name="r128gain",
    version=version,
    author="desbma",
    packages=find_packages(exclude=("tests",)),
    entry_points={"console_scripts": ["r128gain = r128gain:cl_main"]},
    test_suite="tests",
    install_requires=requirements,
    tests_require=test_requirements,
    description="Fast audio loudness scanner & tagger",
    long_description_content_type="text/markdown",
    long_description=readme,
    url="https://github.com/desbma/r128gain",
    download_url="https://github.com/desbma/r128gain/archive/%s.tar.gz" % (version),
    keywords=[
        "audio",
        "loudness",
        "replaygain",
        "replay",
        "gain",
        "r128",
        "tag",
        "opus",
        "normalize",
        "normalization",
        "level",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU Lesser General Public License v2 or later (LGPLv2+)",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Utilities",
    ],
)
