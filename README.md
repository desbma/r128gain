r128gain
========
Fast audio loudness scanner & tagger
------------------------------------

[![Latest version](https://img.shields.io/pypi/v/r128gain.svg?style=flat)](https://pypi.python.org/pypi/r128gain/)
[![Tests status](https://img.shields.io/travis/desbma/r128gain/master.svg?label=tests&style=flat)](https://travis-ci.org/desbma/r128gain)
[![Coverage](https://img.shields.io/coveralls/desbma/r128gain/master.svg?style=flat)](https://coveralls.io/github/desbma/r128gain?branch=master)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/r128gain.svg?style=flat)](https://pypi.python.org/pypi/r128gain/)
[![License](https://img.shields.io/github/license/desbma/r128gain.svg?style=flat)](https://github.com/desbma/r128gain/blob/master/LICENSE)

r128gain is a multi platform command line tool to scan your audio files and tag them with loudness metadata (ReplayGain v2 or Opus R128 gain format), to allow playback of several tracks or albums at a similar loudness level.

**This is beta software, please test and report bugs.**
**Some major features are currently missing (album gain, recursive mode...) and will be implemented before version 1.0 is released.**


## Features

* Use threading to optimally use processor cores resulting in very fast processing
* Supports all common audio file formats: MP3, ACC, Vorbis, Opus, FLAC, WavPack...
* Supports new R128_xxx tag format for Opus files (very few scanners write this tag, athough it is defined in the [Opus standard](https://tools.ietf.org/html/rfc7845#section-5.2))


## Installation

r128gain requires [Python](https://www.python.org/downloads/) >= 3.4.

### From PyPI (with PIP)

Install r128gain using [pip](http://www.pip-installer.org/en/latest/installing.html): `pip3 install r128gain`

### From source

1. If you don't already have it, [install setuptools](https://pypi.python.org/pypi/setuptools#installation-instructions) for Python 3
2. Clone this repository: `git clone https://github.com/desbma/r128gain`
3. Install r128gain: `python3 setup.py install`


## Command line usage

Run `r128gain -h` to get full command line reference.

### Examples

* Scan all MP3 files in `music_directory`: `r128gain music_directory/*`


## License

[LGPLv2](https://www.gnu.org/licenses/old-licenses/lgpl-2.1-standalone.html)
