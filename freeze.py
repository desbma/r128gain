#!/usr/bin/env python3

import os
import re

from cx_Freeze import setup, Executable


with open(os.path.join("r128gain", "__init__.py"), "rt") as f:
  version = re.search("__version__ = \"([^\"]+)\"", f.read()).group(1)

build_exe_options = {"optimize": 0}

setup(name="r128gain",
      version=version,
      author="desbma",
      packages=["r128gain"],
      options={"build_exe": build_exe_options},
      executables=[Executable(os.path.join("r128gain", "__main__.py"),
                              targetName="r128gain.exe")])
