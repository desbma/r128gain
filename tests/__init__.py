#!/usr/bin/env python3

import logging
import os
import random
import shutil
import tempfile
import unittest
import urllib

import mutagen
import requests

import r128gain


def download(url, filepath):
  cache_dir = os.getenv("TEST_DL_CACHE_DIR")
  if cache_dir is not None:
    os.makedirs(cache_dir, exist_ok=True)
    cache_filepath = os.path.join(cache_dir,
                                  os.path.basename(urllib.parse.urlsplit(url).path))
    if os.path.isfile(cache_filepath):
      shutil.copyfile(cache_filepath, filepath)
      return
  with requests.get(url, stream=True) as response, open(filepath, "wb") as file:
    for chunk in response.iter_content(chunk_size=2 ** 14):
      file.write(chunk)
  if cache_dir is not None:
    shutil.copyfile(filepath, cache_filepath)


class TestR128Gain(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.ref_temp_dir = tempfile.TemporaryDirectory()
    vorbis_filepath = os.path.join(cls.ref_temp_dir.name, "f.ogg")
    download("https://upload.wikimedia.org/wikipedia/en/0/09/Opeth_-_Deliverance.ogg",
             vorbis_filepath)
    opus_filepath = os.path.join(cls.ref_temp_dir.name, "f.opus")
    download("https://people.xiph.org/~giles/2012/opus/ehren-paper_lights-64.opus",
             opus_filepath)
    mp3_filepath = os.path.join(cls.ref_temp_dir.name, "f.mp3")
    download("https://allthingsaudio.wikispaces.com/file/view/Shuffle%20for%20K.M.mp3/139190697/Shuffle%20for%20K.M.mp3",
             mp3_filepath)
    m4a_filepath = os.path.join(cls.ref_temp_dir.name, "f.m4a")
    download("https://auphonic.com/media/audio-examples/01.auphonic-demo-unprocessed.m4a",
             m4a_filepath)

  @classmethod
  def tearDownClass(cls):
    cls.ref_temp_dir.cleanup()

  def setUp(self):
    self.temp_dir = tempfile.TemporaryDirectory()
    for src_filename in os.listdir(__class__.ref_temp_dir.name):
      shutil.copy(os.path.join(__class__.ref_temp_dir.name, src_filename), self.temp_dir.name)
    self.vorbis_filepath = os.path.join(self.temp_dir.name, "f.ogg")
    self.opus_filepath = os.path.join(self.temp_dir.name, "f.opus")
    self.mp3_filepath = os.path.join(self.temp_dir.name, "f.mp3")
    self.m4a_filepath = os.path.join(self.temp_dir.name, "f.m4a")

  def tearDown(self):
    self.temp_dir.cleanup()

  def test_scan(self):
    self.assertEqual(r128gain.scan((self.vorbis_filepath,
                                    self.opus_filepath,
                                    self.mp3_filepath,
                                    self.m4a_filepath)),
                     {self.vorbis_filepath: (-7.7, 2.6),
                      self.opus_filepath: (-14.7, None),
                      self.mp3_filepath: (-15.3, -0.1),
                      self.m4a_filepath: (-20.6, 0.1)})

  def test_tag(self):
    loudness = -3.2
    peak = -0.2
    ref_loudness = -18

    for i, delete_tags in zip(range(3), (False, True, False)):
      # i = 0 : add RG tag in existing tags
      # i = 1 : add RG tag with no existing tags
      # i = 2 : overwrites RG tag in existing tags
      with self.subTest(iteration=i + 1, delete_tags=delete_tags):
        if delete_tags:
          for file in (self.vorbis_filepath,
                       self.opus_filepath,
                       self.mp3_filepath,
                       self.m4a_filepath):
            mf = mutagen.File(file)
            mf.delete()
            mf.save()

        if delete_tags:
          mf = mutagen.File(self.vorbis_filepath)
          self.assertNotIn("REPLAYGAIN_TRACK_GAIN", mf)
          self.assertNotIn("REPLAYGAIN_TRACK_PEAK", mf)
        r128gain.tag(self.vorbis_filepath, loudness, peak, ref_loudness=ref_loudness)
        mf = mutagen.File(self.vorbis_filepath)
        self.assertIn("REPLAYGAIN_TRACK_GAIN", mf)
        self.assertEqual(mf["REPLAYGAIN_TRACK_GAIN"], ["-14.80 dB"])
        self.assertIn("REPLAYGAIN_TRACK_PEAK", mf)
        self.assertEqual(mf["REPLAYGAIN_TRACK_PEAK"], ["0.97723722"])

        if delete_tags:
          mf = mutagen.File(self.vorbis_filepath)
          self.assertNotIn("R128_TRACK_GAIN", mf)
        r128gain.tag(self.opus_filepath, loudness, peak, ref_loudness=ref_loudness)
        mf = mutagen.File(self.opus_filepath)
        self.assertIn("R128_TRACK_GAIN", mf)
        self.assertEqual(mf["R128_TRACK_GAIN"], ["-3789"])

        if delete_tags:
          mf = mutagen.File(self.vorbis_filepath)
          self.assertNotIn("TXXX:REPLAYGAIN_TRACK_GAIN", mf)
          self.assertNotIn("TXXX:REPLAYGAIN_TRACK_PEAK", mf)
        r128gain.tag(self.mp3_filepath, loudness, peak, ref_loudness=ref_loudness)
        mf = mutagen.File(self.mp3_filepath)
        self.assertIn("TXXX:REPLAYGAIN_TRACK_GAIN", mf)
        self.assertEqual(mf["TXXX:REPLAYGAIN_TRACK_GAIN"].text, ["-14.80 dB"])
        self.assertIn("TXXX:REPLAYGAIN_TRACK_PEAK", mf)
        self.assertEqual(mf["TXXX:REPLAYGAIN_TRACK_PEAK"].text, ["0.977237"])

        if delete_tags:
          mf = mutagen.File(self.vorbis_filepath)
          self.assertNotIn("----:COM.APPLE.ITUNES:REPLAYGAIN_TRACK_GAIN", mf)
          self.assertNotIn("----:COM.APPLE.ITUNES:REPLAYGAIN_TRACK_PEAK", mf)
        r128gain.tag(self.m4a_filepath, loudness, peak, ref_loudness=ref_loudness)
        mf = mutagen.File(self.m4a_filepath)
        self.assertIn("----:COM.APPLE.ITUNES:REPLAYGAIN_TRACK_GAIN", mf)
        self.assertEqual(len(mf["----:COM.APPLE.ITUNES:REPLAYGAIN_TRACK_GAIN"]), 1)
        self.assertEqual(bytes(mf["----:COM.APPLE.ITUNES:REPLAYGAIN_TRACK_GAIN"][0]).decode(),
                         "-14.80 dB")
        self.assertIn("----:COM.APPLE.ITUNES:REPLAYGAIN_TRACK_PEAK", mf)
        self.assertEqual(len(mf["----:COM.APPLE.ITUNES:REPLAYGAIN_TRACK_PEAK"]), 1)
        self.assertEqual(bytes(mf["----:COM.APPLE.ITUNES:REPLAYGAIN_TRACK_PEAK"][0]).decode(),
                         "0.977237")

  def test_process(self):
    ref_loudness = random.randint(-30, 10)
    r128gain.process((self.vorbis_filepath,
                      self.opus_filepath,
                      self.mp3_filepath,
                      self.m4a_filepath),
                     ref_loudness=ref_loudness)

    mf = mutagen.File(self.vorbis_filepath)
    self.assertIn("REPLAYGAIN_TRACK_GAIN", mf)
    self.assertEqual(mf["REPLAYGAIN_TRACK_GAIN"], ["%.2f dB" % (7.7 + ref_loudness)])
    self.assertIn("REPLAYGAIN_TRACK_PEAK", mf)
    self.assertEqual(mf["REPLAYGAIN_TRACK_PEAK"], ["1.34896288"])

    mf = mutagen.File(self.opus_filepath)
    self.assertIn("R128_TRACK_GAIN", mf)
    self.assertEqual(mf["R128_TRACK_GAIN"], [str(r128gain.float_to_q7dot8(14.7 + ref_loudness))])

    mf = mutagen.File(self.mp3_filepath)
    self.assertIn("TXXX:REPLAYGAIN_TRACK_GAIN", mf)
    self.assertEqual(mf["TXXX:REPLAYGAIN_TRACK_GAIN"].text, ["%.2f dB" % (15.3 + ref_loudness)])
    self.assertIn("TXXX:REPLAYGAIN_TRACK_PEAK", mf)
    self.assertEqual(mf["TXXX:REPLAYGAIN_TRACK_PEAK"].text, ["0.988553"])

    mf = mutagen.File(self.m4a_filepath)
    self.assertIn("----:COM.APPLE.ITUNES:REPLAYGAIN_TRACK_GAIN", mf)
    self.assertEqual(len(mf["----:COM.APPLE.ITUNES:REPLAYGAIN_TRACK_GAIN"]), 1)
    self.assertEqual(bytes(mf["----:COM.APPLE.ITUNES:REPLAYGAIN_TRACK_GAIN"][0]).decode(),
                     "%.2f dB" % (20.6 + ref_loudness))
    self.assertIn("----:COM.APPLE.ITUNES:REPLAYGAIN_TRACK_PEAK", mf)
    self.assertEqual(len(mf["----:COM.APPLE.ITUNES:REPLAYGAIN_TRACK_PEAK"]), 1)
    self.assertEqual(bytes(mf["----:COM.APPLE.ITUNES:REPLAYGAIN_TRACK_PEAK"][0]).decode(),
                     "1.011579")

if __name__ == "__main__":
  # disable logging
  logging.basicConfig(level=logging.CRITICAL + 1)
  #logging.basicConfig(level=logging.DEBUG)

  # run tests
  unittest.main()
