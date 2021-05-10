#!/usr/bin/env python3

""" Package freezing for Windows. """

import os
import re

from cx_Freeze import Executable, setup

with open(os.path.join("r128gain", "__init__.py"), "rt") as f:
    version_match = re.search('__version__ = "([^"]+)"', f.read())
assert version_match is not None
version = version_match.group(1)

build_exe_options = {"optimize": 0, "excludes": ["tkinter"]}

setup(
    name="r128gain",
    version=version,
    author="desbma",
    packages=["r128gain"],
    options={"build_exe": build_exe_options},
    executables=[Executable(os.path.join("r128gain", "__main__.py"), targetName="r128gain.exe")],
)
