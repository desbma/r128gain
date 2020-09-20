#!/usr/bin/env python3

import itertools
import logging
import math
import os
import random
import re
import shutil
import subprocess
import tempfile
import unittest
import unittest.mock
import urllib
import zipfile

import mutagen
import requests

import r128gain
import r128gain.opusgain


IS_TRAVIS = os.getenv("CI") and os.getenv("TRAVIS")


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


@unittest.skipUnless(shutil.which("sox") is not None, "SoX binary is needed")
class TestR128Gain(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.ref_temp_dir = tempfile.TemporaryDirectory()

    vorbis_filepath = os.path.join(cls.ref_temp_dir.name, "f.ogg")
    download("https://upload.wikimedia.org/wikipedia/en/0/09/Opeth_-_Deliverance.ogg",
             vorbis_filepath)

    opus_filepath = os.path.join(cls.ref_temp_dir.name, "f.opus")
    download("https://www.dropbox.com/s/xlp1goezxovlgl4/ehren-paper_lights-64.opus?dl=1",
             opus_filepath)

    cls.ref_temp_dir2 = tempfile.TemporaryDirectory()
    opus_filepath2 = os.path.join(cls.ref_temp_dir2.name, "f2.opus")
    shutil.copyfile(opus_filepath, opus_filepath2)

    mp3_filepath = os.path.join(cls.ref_temp_dir.name, "f.mp3")
    download("http://www.largesound.com/ashborytour/sound/brobob.mp3",
             mp3_filepath)

    invalid_mp3_filepath = os.path.join(cls.ref_temp_dir.name, "invalid.mp3")
    download("https://www.dropbox.com/s/bdm7wyqaj3ij8ci/Death%20Star.mp3?dl=1",
             invalid_mp3_filepath)

    m4a_filepath = os.path.join(cls.ref_temp_dir.name, "f.m4a")
    download("https://auphonic.com/media/audio-examples/01.auphonic-demo-unprocessed.m4a",
             m4a_filepath)

    flac_filepath = os.path.join(cls.ref_temp_dir.name, "f.flac")
    flac_zic_filepath = "%s.zip" % (flac_filepath)
    download("http://helpguide.sony.net/high-res/sample1/v1/data/Sample_HisokanaMizugame_88_2kHz24bit.flac.zip",
             flac_zic_filepath)
    with zipfile.ZipFile(flac_zic_filepath) as zip_file:
      in_filename = zip_file.namelist()[0]
      with zip_file.open(in_filename) as in_file:
        with open(flac_filepath, "wb") as out_file:
          shutil.copyfileobj(in_file, out_file)
    os.remove(flac_zic_filepath)

    flac_filepath = os.path.join(cls.ref_temp_dir.name, "f2.flac")
    flac_zic_filepath = "%s.zip" % (flac_filepath)
    download("https://github.com/desbma/r128gain/files/3006101/snippet_with_high_true_peak.zip",
             flac_zic_filepath)
    with zipfile.ZipFile(flac_zic_filepath) as zip_file:
      in_filename = zip_file.namelist()[0]
      with zip_file.open(in_filename) as in_file:
        with open(flac_filepath, "wb") as out_file:
          shutil.copyfileobj(in_file, out_file)
    os.remove(flac_zic_filepath)

    wv_filepath = os.path.join(cls.ref_temp_dir.name, "f.wv")
    cmd = ("sox", "-R", "-n",
           "-b", "16", "-c", "2", "-r", "44.1k", "-t", "wv", wv_filepath,
           "synth", "30", "sine", "1-5000")
    subprocess.run(cmd, check=True)

    silence_wv_filepath = os.path.join(cls.ref_temp_dir.name, "silence.wv")
    cmd = ("sox", "-R", "-n",
           "-b", "16", "-c", "2", "-r", "44.1k", "-t", "wv", silence_wv_filepath,
           "trim", "0", "10")
    subprocess.run(cmd, check=True)

    silence_mp3_filepath = os.path.join(cls.ref_temp_dir.name, "silence.mp3")
    download("https://www.dropbox.com/s/hnkmioxwu56dgs0/04-There%27s%20No%20Other%20Way.mp3?dl=1",
             silence_mp3_filepath)

  @classmethod
  def tearDownClass(cls):
    cls.ref_temp_dir.cleanup()
    cls.ref_temp_dir2.cleanup()

  def setUp(self):
    self.maxDiff = None

    self.temp_dir = tempfile.TemporaryDirectory()
    for src_filename in os.listdir(__class__.ref_temp_dir.name):
      shutil.copy(os.path.join(__class__.ref_temp_dir.name, src_filename), self.temp_dir.name)

    self.temp_dir2 = tempfile.TemporaryDirectory()
    for src_filename in os.listdir(__class__.ref_temp_dir2.name):
      shutil.copy(os.path.join(__class__.ref_temp_dir2.name, src_filename), self.temp_dir2.name)

    self.vorbis_filepath = os.path.join(self.temp_dir.name, "f.ogg")
    self.opus_filepath = os.path.join(self.temp_dir.name, "f.opus")
    self.opus_filepath2 = os.path.join(self.temp_dir2.name, "f2.opus")
    self.mp3_filepath = os.path.join(self.temp_dir.name, "f.mp3")
    self.invalid_mp3_filepath = os.path.join(self.temp_dir.name, "invalid.mp3")
    self.m4a_filepath = os.path.join(self.temp_dir.name, "f.m4a")
    self.flac_filepath = os.path.join(self.temp_dir.name, "f.flac")
    self.flac_filepath_2 = os.path.join(self.temp_dir.name, "f2.flac")
    self.wv_filepath = os.path.join(self.temp_dir.name, "f.wv")
    self.silence_wv_filepath = os.path.join(self.temp_dir.name, "silence.wv")
    self.silence_mp3_filepath = os.path.join(self.temp_dir.name, "silence.mp3")

    self.ref_levels = {self.vorbis_filepath: (-7.7, 1.0),
                       self.opus_filepath: (-14.7, None),
                       self.mp3_filepath: (-13.9, 0.94281),
                       self.m4a_filepath: (-20.6, 0.711426),
                       self.flac_filepath: (-26.7, 0.232147),
                       self.wv_filepath: (-3.3, 0.705017),
                       0: (-11.4, 1.0)}
    self.ref_levels_2 = self.ref_levels.copy()
    self.ref_levels_2.update({self.flac_filepath_2: (-6.2, 1.0),
                              self.silence_wv_filepath: (-70.0, 0.000031),
                              self.silence_mp3_filepath: (-70.0, 0.0),
                              0: (-10.9, 1.0)})

    # for "loudest" album mode
    self.ref_levels_loudest = self.ref_levels.copy()
    self.ref_levels_loudest.update({0: (-3.3, 1.0)})

    self.ref_levels_loudest_2 = self.ref_levels_2.copy()
    self.ref_levels_loudest_2.update({self.opus_filepath: (-14.7, 1.0),
                                      0: (-3.3, 1.0)})

    self.max_peak_filepath = self.vorbis_filepath

  def tearDown(self):
    self.temp_dir.cleanup()
    self.temp_dir2.cleanup()

  def assertValidGainStr(self, s, places):
    self.assertIsNotNone(re.match("^-?\d{1,2}\.\d{" + str(places) + "} dB$", s))

  def assertGainStrAlmostEqual(self, s, ref):
    val = float(s.split(" ", 1)[0])
    self.assertAlmostEqual(val, ref, delta=0.1)

  def test_float_to_q7dot8(self):
    self.assertEqual(r128gain.float_to_q7dot8(-12.34), -3159)
    self.assertEqual(r128gain.float_to_q7dot8(0.0), 0)
    self.assertEqual(r128gain.float_to_q7dot8(12.34), 3159)

  def test_gain_to_scale(self):
    self.assertAlmostEqual(r128gain.gain_to_scale(-12.34), 0.241546, places=6)
    self.assertAlmostEqual(r128gain.gain_to_scale(0.0), 1.0, places=6)
    self.assertAlmostEqual(r128gain.gain_to_scale(12.34), 4.139997, places=6)

  def test_scale_to_gain(self):
    self.assertEqual(r128gain.scale_to_gain(0.0), -math.inf)
    self.assertAlmostEqual(r128gain.scale_to_gain(0.123456), -18.169756, places=6)
    self.assertAlmostEqual(r128gain.scale_to_gain(1.0), 0.0)
    self.assertAlmostEqual(r128gain.scale_to_gain(1.234567), 1.830293, places=6)

  def test_scan(self):
    for album_gain in (None, "standard", "loudest"):
      # bunch of different files
      filepaths = (self.vorbis_filepath,
                   self.opus_filepath,
                   self.mp3_filepath,
                   self.m4a_filepath,
                   self.flac_filepath,
                   self.flac_filepath_2,
                   self.wv_filepath,
                   self.silence_wv_filepath,
                   self.silence_mp3_filepath)
      ref_levels = self.ref_levels_loudest_2.copy() if album_gain == "loudest" else self.ref_levels_2.copy()
      if not album_gain:
        del ref_levels[r128gain.ALBUM_GAIN_KEY]
      self.assertEqual(r128gain.scan(filepaths,
                                     album_gain=album_gain),
                       ref_levels)

      # opus only files
      filepaths = (self.opus_filepath, self.opus_filepath2)
      ref_levels_opus = {self.opus_filepath: self.ref_levels[self.opus_filepath],
                         self.opus_filepath2: self.ref_levels[self.opus_filepath]}
      if album_gain:
        ref_levels_opus[r128gain.ALBUM_GAIN_KEY] = self.ref_levels[self.opus_filepath]
      self.assertEqual(r128gain.scan(filepaths,
                                     album_gain=album_gain),
                       ref_levels_opus)

      if album_gain:
        # file order should not change results
        filepaths = (self.opus_filepath,  # reduce permutation counts to speed up tests
                     self.m4a_filepath,
                     self.mp3_filepath)
        ref_levels = r128gain.scan(filepaths,
                                   album_gain=album_gain)
        if IS_TRAVIS:
          shuffled_filepaths_len = len(tuple(itertools.permutations(filepaths)))
        for i, shuffled_filepaths in enumerate(itertools.permutations(filepaths), 1):
          if shuffled_filepaths == filepaths:
            continue
          if IS_TRAVIS:
            print("Testing permutation %u/%u..." % (i, shuffled_filepaths_len))
          self.assertEqual(r128gain.scan(shuffled_filepaths,
                                         album_gain=album_gain),
                           ref_levels)

  def test_tag(self):
    loudness = random.randint(-300, -1) / 10
    peak = random.randint(1, 1000) / 1000
    ref_loudness_rg2 = -18
    expected_track_gain_rg2 = ref_loudness_rg2 - loudness
    ref_loudness_opus = -23
    expected_track_gain_opus = ref_loudness_opus - loudness

    files = (self.vorbis_filepath,
             self.opus_filepath,
             self.mp3_filepath,
             self.m4a_filepath,
             self.flac_filepath,
             self.wv_filepath)

    for i, delete_tags in zip(range(3), (False, True, False)):
      # i = 0 : add RG tag in existing tags
      # i = 1 : add RG tag with no existing tags
      # i = 2 : overwrites RG tag in existing tags
      with self.subTest(iteration=i + 1, delete_tags=delete_tags):
        if delete_tags:
          for file in files:
            self.assertEqual(r128gain.has_loudness_tag(file), (True, False))
            mf = mutagen.File(file)
            mf.delete()
            mf.save()
            self.assertEqual(r128gain.has_loudness_tag(file), (False, False))

        if delete_tags:
          mf = mutagen.File(self.vorbis_filepath)
          self.assertNotIn("REPLAYGAIN_TRACK_GAIN", mf)
          self.assertNotIn("REPLAYGAIN_TRACK_PEAK", mf)
        r128gain.tag(self.vorbis_filepath, loudness, peak)
        mf = mutagen.File(self.vorbis_filepath)
        self.assertIsInstance(mf.tags, mutagen._vorbis.VComment)
        self.assertIn("REPLAYGAIN_TRACK_GAIN", mf)
        self.assertEqual(mf["REPLAYGAIN_TRACK_GAIN"],
                         ["%.2f dB" % (expected_track_gain_rg2)])
        self.assertIn("REPLAYGAIN_TRACK_PEAK", mf)
        self.assertEqual(mf["REPLAYGAIN_TRACK_PEAK"], ["%.8f" % (peak)])

        if delete_tags:
          mf = mutagen.File(self.opus_filepath)
          self.assertNotIn("R128_TRACK_GAIN", mf)
        r128gain.tag(self.opus_filepath, loudness, peak)
        mf = mutagen.File(self.opus_filepath)
        self.assertIsInstance(mf.tags, mutagen._vorbis.VComment)
        self.assertIn("R128_TRACK_GAIN", mf)
        self.assertEqual(mf["R128_TRACK_GAIN"], [str(int(round(expected_track_gain_opus * (2 ** 8), 0)))])

        if delete_tags:
          mf = mutagen.File(self.mp3_filepath)
          self.assertNotIn("TXXX:REPLAYGAIN_TRACK_GAIN", mf)
          self.assertNotIn("TXXX:REPLAYGAIN_TRACK_PEAK", mf)
        r128gain.tag(self.mp3_filepath, loudness, peak)
        mf = mutagen.File(self.mp3_filepath)
        self.assertIsInstance(mf.tags, mutagen.id3.ID3)
        self.assertIn("TXXX:REPLAYGAIN_TRACK_GAIN", mf)
        self.assertEqual(mf["TXXX:REPLAYGAIN_TRACK_GAIN"].text,
                         ["%.2f dB" % (expected_track_gain_rg2)])
        self.assertIn("TXXX:REPLAYGAIN_TRACK_PEAK", mf)
        self.assertEqual(mf["TXXX:REPLAYGAIN_TRACK_PEAK"].text, ["%.6f" % (peak)])

        if delete_tags:
          mf = mutagen.File(self.m4a_filepath)
          self.assertNotIn("----:com.apple.iTunes:replaygain_track_gain", mf)
          self.assertNotIn("----:com.apple.iTunes:replaygain_track_peak", mf)
        r128gain.tag(self.m4a_filepath, loudness, peak)
        mf = mutagen.File(self.m4a_filepath)
        self.assertIsInstance(mf.tags, mutagen.mp4.MP4Tags)
        self.assertIn("----:com.apple.iTunes:replaygain_track_gain", mf)
        self.assertEqual(len(mf["----:com.apple.iTunes:replaygain_track_gain"]), 1)
        self.assertEqual(bytes(mf["----:com.apple.iTunes:replaygain_track_gain"][0]).decode(),
                         "%.2f dB" % (expected_track_gain_rg2))
        self.assertIn("----:com.apple.iTunes:replaygain_track_peak", mf)
        self.assertEqual(len(mf["----:com.apple.iTunes:replaygain_track_peak"]), 1)
        self.assertEqual(bytes(mf["----:com.apple.iTunes:replaygain_track_peak"][0]).decode(),
                         "%.6f" % (peak))

        if delete_tags:
          mf = mutagen.File(self.flac_filepath)
          self.assertNotIn("REPLAYGAIN_TRACK_GAIN", mf)
          self.assertNotIn("REPLAYGAIN_TRACK_PEAK", mf)
        r128gain.tag(self.flac_filepath, loudness, peak)
        mf = mutagen.File(self.flac_filepath)
        self.assertIsInstance(mf.tags, mutagen._vorbis.VComment)
        self.assertIn("REPLAYGAIN_TRACK_GAIN", mf)
        self.assertEqual(mf["REPLAYGAIN_TRACK_GAIN"],
                         ["%.2f dB" % (expected_track_gain_rg2)])
        self.assertIn("REPLAYGAIN_TRACK_PEAK", mf)
        self.assertEqual(mf["REPLAYGAIN_TRACK_PEAK"], ["%.8f" % (peak)])

        if delete_tags:
          mf = mutagen.File(self.wv_filepath)
          self.assertNotIn("REPLAYGAIN_TRACK_GAIN", mf)
          self.assertNotIn("REPLAYGAIN_TRACK_PEAK", mf)
        r128gain.tag(self.wv_filepath, loudness, peak)
        mf = mutagen.File(self.wv_filepath)
        self.assertIsInstance(mf.tags, mutagen.apev2.APEv2)
        self.assertIn("REPLAYGAIN_TRACK_GAIN", mf)
        self.assertEqual(str(mf["REPLAYGAIN_TRACK_GAIN"]),
                         "%.2f dB" % (expected_track_gain_rg2))
        self.assertIn("REPLAYGAIN_TRACK_PEAK", mf)
        self.assertEqual(str(mf["REPLAYGAIN_TRACK_PEAK"]), "%.8f" % (peak))

        for file in files:
          self.assertEqual(r128gain.has_loudness_tag(file), (True, False))

    self.assertEqual(r128gain.has_loudness_tag(self.invalid_mp3_filepath), None)

  def test_process(self):
    ref_loudness_rg2 = -18
    ref_loudness_opus = -23

    for album_mode_i, album_mode in enumerate(("standard", "loudest")):

      if album_mode_i > 0:
        self.tearDown()
        self.setUp()

      files = (self.vorbis_filepath,
               self.opus_filepath,
               self.mp3_filepath,
               self.m4a_filepath,
               self.flac_filepath,
               self.wv_filepath)

      for i, skip_tagged in enumerate((True, False, True)):

        if i == 0:
          for file in files:
            self.assertEqual(r128gain.has_loudness_tag(file), (False, False))

        for album_gain in (None, album_mode):
          with self.subTest(i=i, skip_tagged=skip_tagged, album_gain=album_gain), \
                  unittest.mock.patch("r128gain.get_r128_loudness", wraps=r128gain.get_r128_loudness) as get_r128_loudness_mock:

            r128gain.process(files,
                             album_gain=album_gain,
                             skip_tagged=skip_tagged)

            if skip_tagged and (i > 0):
              self.assertEqual(get_r128_loudness_mock.call_count, 0)
            elif album_gain == "standard":
              if skip_tagged:
                self.assertEqual(get_r128_loudness_mock.call_count, 1)
              else:
                self.assertEqual(get_r128_loudness_mock.call_count, 7)
            else:
              self.assertEqual(get_r128_loudness_mock.call_count, 6)

            for file in files:
              self.assertEqual(r128gain.has_loudness_tag(file), (True, album_gain is not None or (i > 0)))

            ref_levels = self.ref_levels_loudest if album_gain == "loudest" else self.ref_levels

            mf = mutagen.File(self.vorbis_filepath)
            self.assertIsInstance(mf.tags, mutagen._vorbis.VComment)
            self.assertIn("REPLAYGAIN_TRACK_GAIN", mf)
            self.assertEqual(mf["REPLAYGAIN_TRACK_GAIN"],
                             ["%.2f dB" % (ref_loudness_rg2 - ref_levels[self.vorbis_filepath][0])])
            self.assertIn("REPLAYGAIN_TRACK_PEAK", mf)
            self.assertEqual(mf["REPLAYGAIN_TRACK_PEAK"],
                             ["%.8f" % (ref_levels[self.vorbis_filepath][1])])
            if album_gain:
              self.assertIn("REPLAYGAIN_ALBUM_GAIN", mf)
              self.assertEqual(mf["REPLAYGAIN_ALBUM_GAIN"],
                               ["%.2f dB" % (ref_loudness_rg2 - ref_levels[r128gain.ALBUM_GAIN_KEY][0])])
              self.assertIn("REPLAYGAIN_ALBUM_PEAK", mf)
              self.assertEqual(mf["REPLAYGAIN_ALBUM_PEAK"],
                               ["%.8f" % (ref_levels[self.max_peak_filepath][1])])
            elif i == 0:
              self.assertNotIn("REPLAYGAIN_ALBUM_GAIN", mf)
              self.assertNotIn("REPLAYGAIN_ALBUM_PEAK", mf)

            mf = mutagen.File(self.opus_filepath)
            self.assertIsInstance(mf.tags, mutagen._vorbis.VComment)
            self.assertIn("R128_TRACK_GAIN", mf)
            self.assertEqual(mf["R128_TRACK_GAIN"],
                             [str(r128gain.float_to_q7dot8(ref_loudness_opus - ref_levels[self.opus_filepath][0]))])
            if album_gain:
              self.assertIn("R128_ALBUM_GAIN", mf)
              self.assertEqual(mf["R128_ALBUM_GAIN"],
                               [str(r128gain.float_to_q7dot8(ref_loudness_opus - ref_levels[r128gain.ALBUM_GAIN_KEY][0]))])
            elif i == 0:
              self.assertNotIn("R128_ALBUM_GAIN", mf)

            mf = mutagen.File(self.mp3_filepath)
            self.assertIsInstance(mf.tags, mutagen.id3.ID3)
            self.assertIn("TXXX:REPLAYGAIN_TRACK_GAIN", mf)
            self.assertEqual(mf["TXXX:REPLAYGAIN_TRACK_GAIN"].text,
                             ["%.2f dB" % (ref_loudness_rg2 - ref_levels[self.mp3_filepath][0])])
            self.assertIn("TXXX:REPLAYGAIN_TRACK_PEAK", mf)
            self.assertEqual(mf["TXXX:REPLAYGAIN_TRACK_PEAK"].text,
                             ["%.6f" % (ref_levels[self.mp3_filepath][1])])
            if album_gain:
              self.assertIn("TXXX:REPLAYGAIN_ALBUM_GAIN", mf)
              self.assertEqual(mf["TXXX:REPLAYGAIN_ALBUM_GAIN"].text,
                               ["%.2f dB" % (ref_loudness_rg2 - ref_levels[r128gain.ALBUM_GAIN_KEY][0])])
              self.assertIn("TXXX:REPLAYGAIN_ALBUM_PEAK", mf)
              self.assertEqual(mf["TXXX:REPLAYGAIN_ALBUM_PEAK"].text,
                               ["%.6f" % (ref_levels[self.max_peak_filepath][1])])
            elif i == 0:
              self.assertNotIn("TXXX:REPLAYGAIN_ALBUM_GAIN", mf)
              self.assertNotIn("TXXX:REPLAYGAIN_ALBUM_GAIN", mf)

            mf = mutagen.File(self.m4a_filepath)
            self.assertIsInstance(mf.tags, mutagen.mp4.MP4Tags)
            self.assertIn("----:com.apple.iTunes:replaygain_track_gain", mf)
            self.assertEqual(len(mf["----:com.apple.iTunes:replaygain_track_gain"]), 1)
            self.assertEqual(bytes(mf["----:com.apple.iTunes:replaygain_track_gain"][0]).decode(),
                             "%.2f dB" % (ref_loudness_rg2 - ref_levels[self.m4a_filepath][0]))
            self.assertIn("----:com.apple.iTunes:replaygain_track_peak", mf)
            self.assertEqual(len(mf["----:com.apple.iTunes:replaygain_track_peak"]), 1)
            self.assertEqual(bytes(mf["----:com.apple.iTunes:replaygain_track_peak"][0]).decode(),
                             "%.6f" % (ref_levels[self.m4a_filepath][1]))
            if album_gain:
              self.assertIn("----:com.apple.iTunes:replaygain_album_gain", mf)
              self.assertEqual(len(mf["----:com.apple.iTunes:replaygain_album_gain"]), 1)
              self.assertEqual(bytes(mf["----:com.apple.iTunes:replaygain_album_gain"][0]).decode(),
                               "%.2f dB" % (ref_loudness_rg2 - ref_levels[r128gain.ALBUM_GAIN_KEY][0]))
              self.assertIn("----:com.apple.iTunes:replaygain_album_peak", mf)
              self.assertEqual(len(mf["----:com.apple.iTunes:replaygain_album_peak"]), 1)
              self.assertEqual(bytes(mf["----:com.apple.iTunes:replaygain_album_peak"][0]).decode(),
                               "%.6f" % (ref_levels[self.max_peak_filepath][1]))
            elif i == 0:
              self.assertNotIn("----:com.apple.iTunes:replaygain_album_gain", mf)
              self.assertNotIn("----:com.apple.iTunes:replaygain_album_peak", mf)

            mf = mutagen.File(self.flac_filepath)
            self.assertIsInstance(mf.tags, mutagen._vorbis.VComment)
            self.assertIn("REPLAYGAIN_TRACK_GAIN", mf)
            self.assertEqual(mf["REPLAYGAIN_TRACK_GAIN"],
                             ["%.2f dB" % (ref_loudness_rg2 - ref_levels[self.flac_filepath][0])])
            self.assertIn("REPLAYGAIN_TRACK_PEAK", mf)
            self.assertEqual(mf["REPLAYGAIN_TRACK_PEAK"],
                             ["%.8f" % (ref_levels[self.flac_filepath][1])])
            if album_gain:
              self.assertIn("REPLAYGAIN_ALBUM_GAIN", mf)
              self.assertEqual(mf["REPLAYGAIN_ALBUM_GAIN"],
                               ["%.2f dB" % (ref_loudness_rg2 - ref_levels[r128gain.ALBUM_GAIN_KEY][0])])
              self.assertIn("REPLAYGAIN_ALBUM_PEAK", mf)
              self.assertEqual(mf["REPLAYGAIN_ALBUM_PEAK"],
                               ["%.8f" % (ref_levels[self.max_peak_filepath][1])])
            elif i == 0:
              self.assertNotIn("REPLAYGAIN_ALBUM_GAIN", mf)
              self.assertNotIn("REPLAYGAIN_ALBUM_PEAK", mf)

            mf = mutagen.File(self.wv_filepath)
            self.assertIsInstance(mf.tags, mutagen.apev2.APEv2)
            self.assertIn("REPLAYGAIN_TRACK_GAIN", mf)
            self.assertEqual(str(mf["REPLAYGAIN_TRACK_GAIN"]),
                             "%.2f dB" % (ref_loudness_rg2 - ref_levels[self.wv_filepath][0]))
            self.assertIn("REPLAYGAIN_TRACK_PEAK", mf)
            self.assertEqual(str(mf["REPLAYGAIN_TRACK_PEAK"]),
                             "%.8f" % (ref_levels[self.wv_filepath][1]))
            if album_gain:
              self.assertIn("REPLAYGAIN_ALBUM_GAIN", mf)
              self.assertEqual(str(mf["REPLAYGAIN_ALBUM_GAIN"]),
                               "%.2f dB" % (ref_loudness_rg2 - ref_levels[r128gain.ALBUM_GAIN_KEY][0]))
              self.assertIn("REPLAYGAIN_ALBUM_PEAK", mf)
              self.assertEqual(str(mf["REPLAYGAIN_ALBUM_PEAK"]),
                               "%.8f" % (ref_levels[self.max_peak_filepath][1]))
            elif i == 0:
              self.assertNotIn("REPLAYGAIN_ALBUM_GAIN", mf)
              self.assertNotIn("REPLAYGAIN_ALBUM_PEAK", mf)

  def test_process_recursive(self):
    ref_loudness_rg2 = -18
    ref_loudness_opus = -23

    for album_mode_i, album_mode in enumerate(("standard", "loudest")):

      if album_mode_i > 0:
        self.tearDown()
        self.setUp()

      os.remove(self.flac_filepath_2)

      album_dir1 = os.path.join(self.temp_dir.name, "a")
      os.mkdir(album_dir1)
      album1_vorbis_filepath = os.path.join(album_dir1,
                                            os.path.basename(self.vorbis_filepath))
      shutil.copyfile(self.vorbis_filepath, album1_vorbis_filepath)
      album1_opus_filepath = os.path.join(album_dir1,
                                          os.path.basename(self.opus_filepath))
      shutil.copyfile(self.opus_filepath, album1_opus_filepath)

      album_dir2 = os.path.join(self.temp_dir.name, "b")
      os.mkdir(album_dir2)
      album2_mp3_filepath = os.path.join(album_dir2,
                                         os.path.basename(self.mp3_filepath))
      shutil.copyfile(self.mp3_filepath, album2_mp3_filepath)
      album2_m4a_filepath = os.path.join(album_dir2,
                                         os.path.basename(self.m4a_filepath))
      shutil.copyfile(self.m4a_filepath, album2_m4a_filepath)
      album2_dummy_filepath = os.path.join(album_dir2, "dummy.txt")
      with open(album2_dummy_filepath, "wb") as f:
        f.write(b"\x00")

      # directory tree is as follows (root is self.temp_dir.name):
      # ├── a
      # │   ├── f.ogg
      # │   └── f.opus
      # ├── b
      # │   ├── dummy.txt
      # │   ├── f.m4a
      # │   └── f.mp3
      # ├── f.flac
      # ├── f.m4a
      # ├── f.mp3
      # ├── f.ogg
      # ├── f.opus
      # └── f.wv

      for i, skip_tagged in enumerate((True, False, True)):

        for album_gain in (None, album_mode):
          with self.subTest(i=i, skip_tagged=skip_tagged, album_gain=album_gain), \
                  unittest.mock.patch("r128gain.get_r128_loudness", wraps=r128gain.get_r128_loudness) as get_r128_loudness_mock:

            r128gain.process_recursive((self.temp_dir.name,),
                                       album_gain=album_gain,
                                       skip_tagged=skip_tagged)

            if skip_tagged and (i > 0):
              self.assertEqual(get_r128_loudness_mock.call_count, 0)
            elif album_gain == "standard":
              if skip_tagged:
                self.assertEqual(get_r128_loudness_mock.call_count, 3)
              else:
                self.assertEqual(get_r128_loudness_mock.call_count, 15)
            else:
              self.assertEqual(get_r128_loudness_mock.call_count, 12)

            # root tmp dir

            if album_gain == "loudest":
              ref_levels = self.ref_levels_loudest
              ref_levels_dir1 = (-7.7, 1.0)
              ref_levels_dir2 = (-13.9, 0.942810)
            else:
              ref_levels = self.ref_levels
              ref_levels_dir1 = (-13, 1.0)
              ref_levels_dir2 = (-17.3, 0.942810)


            mf = mutagen.File(self.vorbis_filepath)
            self.assertIsInstance(mf.tags, mutagen._vorbis.VComment)
            self.assertIn("REPLAYGAIN_TRACK_GAIN", mf)
            self.assertEqual(mf["REPLAYGAIN_TRACK_GAIN"],
                             ["%.2f dB" % (ref_loudness_rg2 - ref_levels[self.vorbis_filepath][0])])
            self.assertIn("REPLAYGAIN_TRACK_PEAK", mf)
            self.assertEqual(mf["REPLAYGAIN_TRACK_PEAK"],
                             ["%.8f" % (ref_levels[self.vorbis_filepath][1])])
            if album_gain:
              self.assertIn("REPLAYGAIN_ALBUM_GAIN", mf)
              self.assertEqual(len(mf["REPLAYGAIN_ALBUM_GAIN"]), 1)
              self.assertValidGainStr(mf["REPLAYGAIN_ALBUM_GAIN"][0], 2)
              self.assertGainStrAlmostEqual(mf["REPLAYGAIN_ALBUM_GAIN"][0],
                                            ref_loudness_rg2 - ref_levels[r128gain.ALBUM_GAIN_KEY][0])
              self.assertIn("REPLAYGAIN_ALBUM_PEAK", mf)
              self.assertEqual(mf["REPLAYGAIN_ALBUM_PEAK"],
                               ["%.8f" % (ref_levels[self.max_peak_filepath][1])])
            elif i == 0:
              self.assertNotIn("REPLAYGAIN_ALBUM_GAIN", mf)
              self.assertNotIn("REPLAYGAIN_ALBUM_PEAK", mf)

            mf = mutagen.File(self.opus_filepath)
            self.assertIsInstance(mf.tags, mutagen._vorbis.VComment)
            self.assertIn("R128_TRACK_GAIN", mf)
            self.assertEqual(mf["R128_TRACK_GAIN"],
                             [str(r128gain.float_to_q7dot8(ref_loudness_opus - ref_levels[self.opus_filepath][0]))])
            if album_gain:
              self.assertIn("R128_ALBUM_GAIN", mf)
              self.assertEqual(len(mf["R128_ALBUM_GAIN"]), 1)
              self.assertIsInstance(mf["R128_ALBUM_GAIN"][0], str)
              ref_q7dot8 = r128gain.float_to_q7dot8(ref_loudness_opus - ref_levels[r128gain.ALBUM_GAIN_KEY][0])
              self.assertTrue(ref_q7dot8 - 30 <= int(mf["R128_ALBUM_GAIN"][0]) <= ref_q7dot8 + 30)
            elif i == 0:
              self.assertNotIn("R128_ALBUM_GAIN", mf)

            mf = mutagen.File(self.mp3_filepath)
            self.assertIsInstance(mf.tags, mutagen.id3.ID3)
            self.assertIn("TXXX:REPLAYGAIN_TRACK_GAIN", mf)
            self.assertEqual(mf["TXXX:REPLAYGAIN_TRACK_GAIN"].text,
                             ["%.2f dB" % (ref_loudness_rg2 - ref_levels[self.mp3_filepath][0])])
            self.assertIn("TXXX:REPLAYGAIN_TRACK_PEAK", mf)
            self.assertEqual(mf["TXXX:REPLAYGAIN_TRACK_PEAK"].text,
                             ["%.6f" % (ref_levels[self.mp3_filepath][1])])
            if album_gain:
              self.assertIn("TXXX:REPLAYGAIN_ALBUM_GAIN", mf)
              self.assertEqual(len(mf["TXXX:REPLAYGAIN_ALBUM_GAIN"].text), 1)
              self.assertValidGainStr(mf["TXXX:REPLAYGAIN_ALBUM_GAIN"].text[0], 2)
              self.assertGainStrAlmostEqual(mf["TXXX:REPLAYGAIN_ALBUM_GAIN"].text[0],
                                            ref_loudness_rg2 - ref_levels[r128gain.ALBUM_GAIN_KEY][0])
              self.assertIn("TXXX:REPLAYGAIN_ALBUM_PEAK", mf)
              self.assertEqual(mf["TXXX:REPLAYGAIN_ALBUM_PEAK"].text,
                               ["%.6f" % (ref_levels[self.max_peak_filepath][1])])
            elif i == 0:
              self.assertNotIn("TXXX:REPLAYGAIN_ALBUM_GAIN", mf)
              self.assertNotIn("TXXX:REPLAYGAIN_ALBUM_GAIN", mf)

            mf = mutagen.File(self.m4a_filepath)
            self.assertIsInstance(mf.tags, mutagen.mp4.MP4Tags)
            self.assertIn("----:com.apple.iTunes:replaygain_track_gain", mf)
            self.assertEqual(len(mf["----:com.apple.iTunes:replaygain_track_gain"]), 1)
            self.assertEqual(bytes(mf["----:com.apple.iTunes:replaygain_track_gain"][0]).decode(),
                             "%.2f dB" % (ref_loudness_rg2 - ref_levels[self.m4a_filepath][0]))
            self.assertIn("----:com.apple.iTunes:replaygain_track_peak", mf)
            self.assertEqual(len(mf["----:com.apple.iTunes:replaygain_track_peak"]), 1)
            self.assertEqual(bytes(mf["----:com.apple.iTunes:replaygain_track_peak"][0]).decode(),
                             "%.6f" % (ref_levels[self.m4a_filepath][1]))
            if album_gain:
              self.assertIn("----:com.apple.iTunes:replaygain_album_gain", mf)
              self.assertEqual(len(mf["----:com.apple.iTunes:replaygain_album_gain"]), 1)
              self.assertValidGainStr(bytes(mf["----:com.apple.iTunes:replaygain_album_gain"][0]).decode(), 2)
              self.assertGainStrAlmostEqual(bytes(mf["----:com.apple.iTunes:replaygain_album_gain"][0]).decode(),
                                            ref_loudness_rg2 - ref_levels[r128gain.ALBUM_GAIN_KEY][0])
              self.assertIn("----:com.apple.iTunes:replaygain_album_peak", mf)
              self.assertEqual(len(mf["----:com.apple.iTunes:replaygain_album_peak"]), 1)
              self.assertEqual(bytes(mf["----:com.apple.iTunes:replaygain_album_peak"][0]).decode(),
                               "%.6f" % (ref_levels[self.max_peak_filepath][1]))
            elif i == 0:
              self.assertNotIn("----:com.apple.iTunes:replaygain_album_gain", mf)
              self.assertNotIn("----:com.apple.iTunes:replaygain_album_peak", mf)

            mf = mutagen.File(self.flac_filepath)
            self.assertIsInstance(mf.tags, mutagen._vorbis.VComment)
            self.assertIn("REPLAYGAIN_TRACK_GAIN", mf)
            self.assertEqual(mf["REPLAYGAIN_TRACK_GAIN"],
                             ["%.2f dB" % (ref_loudness_rg2 - ref_levels[self.flac_filepath][0])])
            self.assertIn("REPLAYGAIN_TRACK_PEAK", mf)
            self.assertEqual(mf["REPLAYGAIN_TRACK_PEAK"],
                             ["%.8f" % (ref_levels[self.flac_filepath][1])])
            if album_gain:
              self.assertIn("REPLAYGAIN_ALBUM_GAIN", mf)
              self.assertEqual(len(mf["REPLAYGAIN_ALBUM_GAIN"]), 1)
              self.assertValidGainStr(mf["REPLAYGAIN_ALBUM_GAIN"][0], 2)
              self.assertGainStrAlmostEqual(mf["REPLAYGAIN_ALBUM_GAIN"][0],
                                            ref_loudness_rg2 - ref_levels[r128gain.ALBUM_GAIN_KEY][0])
              self.assertIn("REPLAYGAIN_ALBUM_PEAK", mf)
              self.assertEqual(mf["REPLAYGAIN_ALBUM_PEAK"],
                               ["%.8f" % (ref_levels[self.max_peak_filepath][1])])
            elif i == 0:
              self.assertNotIn("REPLAYGAIN_ALBUM_GAIN", mf)
              self.assertNotIn("REPLAYGAIN_ALBUM_PEAK", mf)

            mf = mutagen.File(self.wv_filepath)
            self.assertIsInstance(mf.tags, mutagen.apev2.APEv2)
            self.assertIn("REPLAYGAIN_TRACK_GAIN", mf)
            self.assertEqual(str(mf["REPLAYGAIN_TRACK_GAIN"]),
                             "%.2f dB" % (ref_loudness_rg2 - ref_levels[self.wv_filepath][0]))
            self.assertIn("REPLAYGAIN_TRACK_PEAK", mf)
            self.assertEqual(str(mf["REPLAYGAIN_TRACK_PEAK"]),
                             "%.8f" % (ref_levels[self.wv_filepath][1]))
            if album_gain:
              self.assertIn("REPLAYGAIN_ALBUM_GAIN", mf)
              self.assertValidGainStr(str(mf["REPLAYGAIN_ALBUM_GAIN"]), 2)
              self.assertGainStrAlmostEqual(str(mf["REPLAYGAIN_ALBUM_GAIN"]),
                                            ref_loudness_rg2 - ref_levels[r128gain.ALBUM_GAIN_KEY][0])
              self.assertIn("REPLAYGAIN_ALBUM_PEAK", mf)
              self.assertEqual(str(mf["REPLAYGAIN_ALBUM_PEAK"]),
                               "%.8f" % (ref_levels[self.max_peak_filepath][1]))
            elif i == 0:
              self.assertNotIn("REPLAYGAIN_ALBUM_GAIN", mf)
              self.assertNotIn("REPLAYGAIN_ALBUM_PEAK", mf)

            # dir 1

            mf = mutagen.File(album1_vorbis_filepath)
            self.assertIsInstance(mf.tags, mutagen._vorbis.VComment)
            self.assertIn("REPLAYGAIN_TRACK_GAIN", mf)
            self.assertEqual(mf["REPLAYGAIN_TRACK_GAIN"],
                             ["%.2f dB" % (ref_loudness_rg2 - ref_levels[self.vorbis_filepath][0])])
            self.assertIn("REPLAYGAIN_TRACK_PEAK", mf)
            self.assertEqual(mf["REPLAYGAIN_TRACK_PEAK"],
                             ["%.8f" % (ref_levels[self.vorbis_filepath][1])])
            if album_gain:
              self.assertIn("REPLAYGAIN_ALBUM_GAIN", mf)
              self.assertEqual(mf["REPLAYGAIN_ALBUM_GAIN"],
                               ["%.2f dB" % (ref_loudness_rg2 - ref_levels_dir1[0])])
              self.assertIn("REPLAYGAIN_ALBUM_PEAK", mf)
              self.assertEqual(mf["REPLAYGAIN_ALBUM_PEAK"],
                               ["%.8f" % (ref_levels_dir1[1])])
            elif i == 0:
              self.assertNotIn("REPLAYGAIN_ALBUM_GAIN", mf)
              self.assertNotIn("REPLAYGAIN_ALBUM_PEAK", mf)

            mf = mutagen.File(album1_opus_filepath)
            self.assertIsInstance(mf.tags, mutagen._vorbis.VComment)
            self.assertIn("R128_TRACK_GAIN", mf)
            self.assertEqual(mf["R128_TRACK_GAIN"],
                             [str(r128gain.float_to_q7dot8(ref_loudness_opus - ref_levels[self.opus_filepath][0]))])
            if album_gain:
              self.assertIn("R128_ALBUM_GAIN", mf)
              self.assertEqual(mf["R128_ALBUM_GAIN"],
                               [str(r128gain.float_to_q7dot8(ref_loudness_opus - ref_levels_dir1[0]))])
            elif i == 0:
              self.assertNotIn("R128_ALBUM_GAIN", mf)

            # dir 2

            mf = mutagen.File(album2_mp3_filepath)
            self.assertIsInstance(mf.tags, mutagen.id3.ID3)
            self.assertIn("TXXX:REPLAYGAIN_TRACK_GAIN", mf)
            self.assertEqual(mf["TXXX:REPLAYGAIN_TRACK_GAIN"].text,
                             ["%.2f dB" % (ref_loudness_rg2 - ref_levels[self.mp3_filepath][0])])
            self.assertIn("TXXX:REPLAYGAIN_TRACK_PEAK", mf)
            self.assertEqual(mf["TXXX:REPLAYGAIN_TRACK_PEAK"].text,
                             ["%.6f" % (ref_levels[self.mp3_filepath][1])])
            if album_gain:
              self.assertIn("TXXX:REPLAYGAIN_ALBUM_GAIN", mf)
              self.assertEqual(mf["TXXX:REPLAYGAIN_ALBUM_GAIN"].text,
                               ["%.2f dB" % (ref_loudness_rg2 - ref_levels_dir2[0])])
              self.assertIn("TXXX:REPLAYGAIN_ALBUM_PEAK", mf)
              self.assertEqual(mf["TXXX:REPLAYGAIN_ALBUM_PEAK"].text,
                               ["%.6f" % (ref_levels_dir2[1])])
            elif i == 0:
              self.assertNotIn("TXXX:REPLAYGAIN_ALBUM_GAIN", mf)
              self.assertNotIn("TXXX:REPLAYGAIN_ALBUM_GAIN", mf)

            mf = mutagen.File(album2_m4a_filepath)
            self.assertIsInstance(mf.tags, mutagen.mp4.MP4Tags)
            self.assertIn("----:com.apple.iTunes:replaygain_track_gain", mf)
            self.assertEqual(len(mf["----:com.apple.iTunes:replaygain_track_gain"]), 1)
            self.assertEqual(bytes(mf["----:com.apple.iTunes:replaygain_track_gain"][0]).decode(),
                             "%.2f dB" % (ref_loudness_rg2 - ref_levels[self.m4a_filepath][0]))
            self.assertIn("----:com.apple.iTunes:replaygain_track_peak", mf)
            self.assertEqual(len(mf["----:com.apple.iTunes:replaygain_track_peak"]), 1)
            self.assertEqual(bytes(mf["----:com.apple.iTunes:replaygain_track_peak"][0]).decode(),
                             "%.6f" % (ref_levels[self.m4a_filepath][1]))
            if album_gain:
              self.assertIn("----:com.apple.iTunes:replaygain_album_gain", mf)
              self.assertEqual(len(mf["----:com.apple.iTunes:replaygain_album_gain"]), 1)
              self.assertEqual(bytes(mf["----:com.apple.iTunes:replaygain_album_gain"][0]).decode(),
                               "%.2f dB" % (ref_loudness_rg2 - ref_levels_dir2[0]))
              self.assertIn("----:com.apple.iTunes:replaygain_album_peak", mf)
              self.assertEqual(len(mf["----:com.apple.iTunes:replaygain_album_peak"]), 1)
              self.assertEqual(bytes(mf["----:com.apple.iTunes:replaygain_album_peak"][0]).decode(),
                               "%.6f" % (ref_levels_dir2[1]))
            elif i == 0:
              self.assertNotIn("----:com.apple.iTunes:replaygain_album_gain", mf)
              self.assertNotIn("----:com.apple.iTunes:replaygain_album_peak", mf)

  def test_oggopus_output_gain(self):
    # non opus formats should not be parsed successfully
    for filepath in (self.vorbis_filepath,
                     self.mp3_filepath,
                     self.m4a_filepath,
                     self.flac_filepath,
                     self.wv_filepath):
      with open(filepath, "r+b") as f:
        with self.assertRaises(ValueError):
          r128gain.opusgain.parse_oggopus_output_gain(f)

    with open(self.opus_filepath, "r+b") as f:
      self.assertEqual(r128gain.opusgain.parse_oggopus_output_gain(f), 0)

      new_gain = random.randint(-32768, 32767)
      r128gain.opusgain.write_oggopus_output_gain(f, new_gain)

    with open(self.opus_filepath, "rb") as f:
      self.assertEqual(r128gain.opusgain.parse_oggopus_output_gain(f), new_gain)

  def test_tag_oggopus_output_gain(self):
    ref_loudness_opus = -23
    ref_track_loudness = self.ref_levels[self.opus_filepath][0]
    track_to_album_gain_delta = 1.5
    ref_album_loudness = ref_track_loudness + track_to_album_gain_delta

    opus_filepath = shutil.copyfile(self.opus_filepath,
                                    ".".join((os.path.splitext(self.opus_filepath)[0], "2" ".opus")))
    r128gain.tag(opus_filepath, ref_track_loudness, None, opus_output_gain=True)
    mf = mutagen.File(opus_filepath)
    self.assertIsInstance(mf.tags, mutagen._vorbis.VComment)
    self.assertIn("R128_TRACK_GAIN", mf)
    self.assertEqual(mf["R128_TRACK_GAIN"], ["0"])
    self.assertNotIn("R128_ALBUM_GAIN", mf)

    expected_output_gain = round((ref_loudness_opus - ref_track_loudness) * (2 ** 8))
    with open(opus_filepath, "rb") as f:
      self.assertEqual(r128gain.opusgain.parse_oggopus_output_gain(f), expected_output_gain)

    self.assertEqual(r128gain.scan([opus_filepath]),
                     {opus_filepath: (float(ref_loudness_opus), None)})

    opus_filepath = shutil.copyfile(self.opus_filepath,
                                    ".".join((os.path.splitext(self.opus_filepath)[0], "2" ".opus")))
    r128gain.tag(opus_filepath,
                 None,
                 None,
                 album_loudness=ref_album_loudness,
                 opus_output_gain=True)
    mf = mutagen.File(opus_filepath)
    self.assertIsInstance(mf.tags, mutagen._vorbis.VComment)
    self.assertNotIn("R128_TRACK_GAIN", mf)
    self.assertIn("R128_ALBUM_GAIN", mf)
    self.assertEqual(mf["R128_ALBUM_GAIN"], ["0"])

    expected_output_gain = round((ref_loudness_opus - ref_album_loudness) * (2 ** 8))
    with open(opus_filepath, "rb") as f:
      self.assertEqual(r128gain.opusgain.parse_oggopus_output_gain(f), expected_output_gain)

    self.assertEqual(r128gain.scan([opus_filepath]),
                     {opus_filepath: (float(ref_loudness_opus - track_to_album_gain_delta), None)})

    opus_filepath = shutil.copyfile(self.opus_filepath,
                                    ".".join((os.path.splitext(self.opus_filepath)[0], "2" ".opus")))
    r128gain.tag(opus_filepath,
                 ref_track_loudness,
                 None,
                 album_loudness=ref_album_loudness,
                 opus_output_gain=True)
    mf = mutagen.File(opus_filepath)
    self.assertIsInstance(mf.tags, mutagen._vorbis.VComment)
    self.assertIn("R128_TRACK_GAIN", mf)
    self.assertEqual(mf["R128_TRACK_GAIN"], [str(r128gain.float_to_q7dot8(track_to_album_gain_delta))])
    self.assertIn("R128_ALBUM_GAIN", mf)
    self.assertEqual(mf["R128_ALBUM_GAIN"], ["0"])

    expected_output_gain = round((ref_loudness_opus - ref_album_loudness) * (2 ** 8))
    with open(opus_filepath, "rb") as f:
      self.assertEqual(r128gain.opusgain.parse_oggopus_output_gain(f), expected_output_gain)

    self.assertEqual(r128gain.scan([opus_filepath]),
                     {opus_filepath: (float(ref_loudness_opus - track_to_album_gain_delta), None)})


if __name__ == "__main__":
  # disable logging
  logging.basicConfig(level=logging.CRITICAL + 1)
  #logging.basicConfig(level=logging.DEBUG)

  # run tests
  unittest.main()
