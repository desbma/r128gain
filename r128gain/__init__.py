#!/usr/bin/env python3

""" Scan audio files and tag them with ReplayGain/R128 loudness metadata. """

__version__ = "1.0.3"
__author__ = "desbma"
__license__ = "LGPLv2"

import argparse
import collections
import concurrent.futures
import contextlib
import functools
import logging
import math
import operator
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading

import ffmpeg
import mutagen
import tqdm

import r128gain.colored_logging as colored_logging
import r128gain.tqdm_logging as tqdm_logging
import r128gain.opusgain as opusgain


try:
  # Python >= 3.8
  cmd_to_string = shlex.join
except AttributeError:
  cmd_to_string = subprocess.list2cmdline


AUDIO_EXTENSIONS = frozenset(("flac", "ogg", "opus", "m4a", "mp3", "mpc", "tta",
                              "wv"))
RG2_REF_R128_LOUDNESS_DBFS = -18
OPUS_REF_R128_LOUDNESS_DBFS = -23
ALBUM_GAIN_KEY = 0
try:
  OPTIMAL_THREAD_COUNT = len(os.sched_getaffinity(0))
except AttributeError:
  OPTIMAL_THREAD_COUNT = os.cpu_count()


def logger():
  return logging.getLogger("r128gain")


@contextlib.contextmanager
def dynamic_tqdm(*tqdm_args, **tqdm_kwargs):
  """ Context manager that returns a tqdm object or None depending on context. """
  with contextlib.ExitStack() as cm:
    if sys.stderr.isatty() and logger().isEnabledFor(logging.INFO):
      progress = cm.enter_context(tqdm.tqdm(*tqdm_args, **tqdm_kwargs))
      cm.enter_context(tqdm_logging.redirect_logging(progress))
    else:
      progress = None
    yield progress


def is_audio_filepath(filepath):
  """ Return True if filepath is a supported audio file. """
  # TODO more robust way to identify audio files? (open with mutagen?)
  return os.path.splitext(filepath)[-1].lstrip(".").lower() in AUDIO_EXTENSIONS


@functools.lru_cache()
def get_ffmpeg_lib_versions(ffmpeg_path=None):
  """ Get FFmpeg library versions as 32 bit integers, with same format as sys.hexversion.

  Example: 0x3040100 for FFmpeg 3.4.1
  """
  r = collections.OrderedDict()
  cmd = (ffmpeg_path or "ffmpeg", "-version")
  output = subprocess.run(cmd,
                          check=True,
                          stdout=subprocess.PIPE,
                          universal_newlines=True).stdout
  output = output.splitlines()
  lib_version_regex = re.compile(r"^\s*(lib[a-z]+)\s+([0-9]+).\s*([0-9]+).\s*([0-9]+)\s+")
  for line in output:
    match = lib_version_regex.search(line)
    if match:
      lib_name, *lib_version = match.group(1, 2, 3, 4)
      int_lib_version = 0
      for i, d in enumerate(map(int, reversed(lib_version)), 1):
        int_lib_version |= d << (8 * i)
      r[lib_name] = int_lib_version
  return r


def get_r128_loudness(audio_filepaths, *, calc_peak=True, enable_ffmpeg_threading=True, ffmpeg_path=None,
                      start_evt=None):
  """ Get R128 loudness loudness level and sample peak. """
  if start_evt is not None:
    start_evt.wait()

  logger().info("Analyzing loudness of file%s %s..." % ("s" if (len(audio_filepaths) > 1) else "",
                                                        ", ".join(repr(audio_filepath) for audio_filepath in audio_filepaths)))

  # build command line
  ffmpeg_inputs = []
  if not enable_ffmpeg_threading:
    additional_ffmpeg_args = {"threads": 1}  # single decoding thread
  else:
    additional_ffmpeg_args = dict()
  for audio_filepath in audio_filepaths:
    ffmpeg_input = ffmpeg.input(audio_filepath, **additional_ffmpeg_args).audio
    ffmpeg_inputs.append(ffmpeg_input)

  output_streams = []
  ffmpeg_r128_streams = []
  for ffmpeg_input in ffmpeg_inputs:
    if calc_peak:
      split_streams = ffmpeg_input.filter_multi_output("asplit", outputs=2)
      ffmpeg_rg_stream, ffmpeg_r128_stream = split_streams[0], split_streams[1]
      ffmpeg_rg_stream = ffmpeg_rg_stream.filter("aformat",
                                                 sample_fmts="s16",
                                                 channel_layouts="stereo")
      ffmpeg_rg_stream = ffmpeg_rg_stream.filter("replaygain")
      output_streams.append(ffmpeg_rg_stream)
    else:
      ffmpeg_r128_stream = ffmpeg_input
    ffmpeg_r128_stream = ffmpeg_r128_stream.filter("aformat",
                                                   sample_fmts="s16",
                                                   sample_rates="48000",
                                                   channel_layouts="stereo")
    ffmpeg_r128_stream = ffmpeg_r128_stream.filter("afifo")  # needed for FFmpeg < 4.1
    ffmpeg_r128_streams.append(ffmpeg_r128_stream)

  if len(audio_filepaths) > 1:
    ffmpeg_r128_merged = ffmpeg.concat(*ffmpeg_r128_streams, n=len(ffmpeg_r128_streams), v=0, a=1)
  else:
    ffmpeg_r128_merged = ffmpeg_r128_streams[0]
  ffmpeg_r128_merged = ffmpeg_r128_merged.filter("ebur128", framelog="verbose")
  output_streams.append(ffmpeg_r128_merged)

  if (get_ffmpeg_lib_versions()["libavfilter"] >= 0x06526400) and (not enable_ffmpeg_threading):
    additional_ffmpeg_args = {"filter_complex_threads": 1}  # single filter thread
  else:
    additional_ffmpeg_args = dict()
  cmd = ffmpeg.compile(ffmpeg.output(*output_streams,
                                     os.devnull,
                                     **additional_ffmpeg_args,
                                     f="null").global_args("-hide_banner", "-nostats"),
                       cmd=ffmpeg_path or "ffmpeg")

  # run
  logger().debug(cmd_to_string(cmd))
  output = subprocess.run(cmd,
                          check=True,
                          stdin=subprocess.DEVNULL,
                          stderr=subprocess.PIPE).stderr
  output = output.decode("utf-8", errors="replace").splitlines()

  if calc_peak:
    # parse replaygain filter output
    sample_peaks = []
    for line in reversed(output):
      if line.startswith("[Parsed_replaygain_") and ("] track_peak = " in line):
        sample_peaks.append(float(line.rsplit("=", 1)[1]))
        if len(sample_peaks) == len(audio_filepaths):
          break
    sample_peak = max(sample_peaks)
  else:
    sample_peak = None

  # parse r128 filter output
  for i in reversed(range(len(output))):
    line = output[i]
    if line.startswith("[Parsed_ebur128_") and line.endswith("Summary:"):
      break
  output = filter(lambda x: x and not x.startswith("[Parsed_replaygain_"),
                  map(str.strip, output[i:]))
  r128_stats = dict(tuple(map(str.strip, line.split(":", 1))) for line in output if not line.endswith(":"))
  r128_stats = {k: float(v.split(" ", 1)[0]) for k, v in r128_stats.items()}

  return r128_stats["I"], sample_peak


def scan(audio_filepaths, *, album_gain=False, skip_tagged=False, thread_count=None, ffmpeg_path=None, executor=None,
         progress=None, boxed_error_count=None, start_evt=None):
  """ Analyze files, and return a dictionary of filepath to loudness metadata or filepath to future if executor is not None. """
  r128_data = {}

  with contextlib.ExitStack() as cm:
    if executor is None:
      if thread_count is None:
        thread_count = OPTIMAL_THREAD_COUNT
      enable_ffmpeg_threading = thread_count > (len(audio_filepaths) + int(album_gain))
      executor = cm.enter_context(concurrent.futures.ThreadPoolExecutor(max_workers=thread_count))
      asynchronous = False
    else:
      enable_ffmpeg_threading = False
      asynchronous = True

    loudness_tags = tuple(map(has_loudness_tag, audio_filepaths))

    # remove invalid files
    audio_filepaths = tuple(audio_filepath for (audio_filepath,
                                                has_tags) in zip(audio_filepaths,
                                                                 loudness_tags) if has_tags is not None)
    loudness_tags = tuple(filter(None, loudness_tags))

    futures = {}
    if album_gain:
      if skip_tagged and all(map(operator.itemgetter(1), loudness_tags)):
        logger().info("All files already have an album gain tag, skipping album gain scan")
      elif audio_filepaths:
        calc_album_peak = any(map(lambda x: os.path.splitext(x)[-1].lower() != ".opus",
                                  audio_filepaths))
        future = executor.submit(get_r128_loudness,
                                 audio_filepaths,
                                 calc_peak=calc_album_peak,
                                 enable_ffmpeg_threading=enable_ffmpeg_threading,
                                 ffmpeg_path=ffmpeg_path,
                                 start_evt=start_evt)
        futures[future] = ALBUM_GAIN_KEY
    for audio_filepath in audio_filepaths:
      if skip_tagged and has_loudness_tag(audio_filepath)[0]:
        logger().info("File %r already has a track gain tag, skipping track gain scan" % (audio_filepath))
        # create dummy future
        future = executor.submit(lambda: None)
      else:
        if os.path.splitext(audio_filepath)[-1].lower() == ".opus":
          # http://www.rfcreader.com/#rfc7845_line1060
          calc_peak = False
        else:
          calc_peak = True
        future = executor.submit(get_r128_loudness,
                                 (audio_filepath,),
                                 calc_peak=calc_peak,
                                 enable_ffmpeg_threading=enable_ffmpeg_threading,
                                 ffmpeg_path=ffmpeg_path,
                                 start_evt=start_evt)
      futures[future] = audio_filepath

    if asynchronous:
      return futures

    while futures:
      done_futures, _ = concurrent.futures.wait(futures,
                                                return_when=concurrent.futures.FIRST_COMPLETED)

      for done_future in done_futures:
        audio_filepath = futures[done_future]
        try:
          result = done_future.result()
        except Exception as e:
          if audio_filepath == ALBUM_GAIN_KEY:
            logger().warning("Failed to analyze files %s: %s %s" % (", ".join(repr(audio_filepath) for audio_filepath in audio_filepaths),
                                                                    e.__class__.__qualname__,
                                                                    e))
          else:
            logger().warning("Failed to analyze file %r: %s %s" % (audio_filepath,
                                                                   e.__class__.__qualname__,
                                                                   e))
          if boxed_error_count is not None:
            boxed_error_count[0] += 1
        else:
          if result is not None:  # track/album gain was not skipped
            r128_data[audio_filepath] = result

        # update progress bar
        if progress is not None:
          progress.update(1)

        del futures[done_future]

  return r128_data


def float_to_q7dot8(f):
  """ Encode float f to a fixed point Q7.8 integer. """
  # https://en.wikipedia.org/wiki/Q_(number_format)#Float_to_Q
  return int(round(f * (2 ** 8), 0))


def gain_to_scale(gain):
  """ Convert a gain value in dBFS to a float where 1.0 is 0 dBFS. """
  return 10 ** (gain / 20)


def scale_to_gain(scale):
  """ Convert a float value to a gain where 0 dBFS is 1.0. """
  if scale == 0:
    return -math.inf
  return 20 * math.log10(scale)


def tag(filepath, loudness, peak, *,
        album_loudness=None, album_peak=None, opus_output_gain=False, mtime_second_offset=None):
  """ Tag audio file with loudness metadata. """
  assert((loudness is not None) or (album_loudness is not None))

  if peak is not None:
    assert(0 <= peak <= 1.0)
    if album_peak is not None:
      assert(0 <= album_peak <= 1.0)

  logger().info("Tagging file %r" % (filepath))
  original_mtime = os.path.getmtime(filepath)
  mf = mutagen.File(filepath)
  if (mf is not None) and (mf.tags is None) and isinstance(mf, mutagen.trueaudio.TrueAudio):
    # TTA can have ID3 or APE tags, try to use APE if we already have some APE tags and no ID3
    try:
      mf_ape = mutagen.apev2.APEv2File(filepath)
    except Exception:
      pass
    else:
      if mf_ape.tags is not None:
        mf = mf_ape
  if (mf is not None) and (mf.tags is None):
    mf.add_tags()

  if (isinstance(mf.tags, mutagen.id3.ID3) or
          isinstance(mf, mutagen.id3.ID3FileType)):
    # http://wiki.hydrogenaud.io/index.php?title=ReplayGain_2.0_specification#ID3v2
    if loudness is not None:
      mf.tags.add(mutagen.id3.TXXX(encoding=mutagen.id3.Encoding.LATIN1,
                                   desc="REPLAYGAIN_TRACK_GAIN",
                                   text="%.2f dB" % (RG2_REF_R128_LOUDNESS_DBFS - loudness)))
      mf.tags.add(mutagen.id3.TXXX(encoding=mutagen.id3.Encoding.LATIN1,
                                   desc="REPLAYGAIN_TRACK_PEAK",
                                   text="%.6f" % (peak)))
    if album_loudness is not None:
      mf.tags.add(mutagen.id3.TXXX(encoding=mutagen.id3.Encoding.LATIN1,
                                   desc="REPLAYGAIN_ALBUM_GAIN",
                                   text="%.2f dB" % (RG2_REF_R128_LOUDNESS_DBFS - album_loudness)))
      mf.tags.add(mutagen.id3.TXXX(encoding=mutagen.id3.Encoding.LATIN1,
                                   desc="REPLAYGAIN_ALBUM_PEAK",
                                   text="%.6f" % (album_peak)))
    # other legacy formats:
    # http://wiki.hydrogenaud.io/index.php?title=ReplayGain_legacy_metadata_formats#ID3v2_RGAD
    # http://wiki.hydrogenaud.io/index.php?title=ReplayGain_legacy_metadata_formats#ID3v2_RVA2

  elif isinstance(mf, mutagen.oggopus.OggOpus):
    if opus_output_gain:
      with open(filepath, "r+b") as file:
        current_output_gain = opusgain.parse_oggopus_output_gain(file)
        if album_loudness is not None:
          # write header relative to album level
          output_gain_loudness = album_loudness
        else:
          output_gain_loudness = loudness
        new_output_gain = current_output_gain + float_to_q7dot8(OPUS_REF_R128_LOUDNESS_DBFS - output_gain_loudness)
        opusgain.write_oggopus_output_gain(file, new_output_gain)

      # now that the output gain header is written, we will write the R128 tag for the new loudness
      if album_loudness is not None:
        if loudness is not None:
          loudness = OPUS_REF_R128_LOUDNESS_DBFS - (album_loudness - loudness)
        album_loudness = OPUS_REF_R128_LOUDNESS_DBFS
      else:
        loudness = OPUS_REF_R128_LOUDNESS_DBFS

    # https://wiki.xiph.org/OggOpus#Comment_Header
    if loudness is not None:
      q78 = float_to_q7dot8(OPUS_REF_R128_LOUDNESS_DBFS - loudness)
      assert(-32768 <= q78 <= 32767)
      mf["R128_TRACK_GAIN"] = str(q78)
    if album_loudness is not None:
      q78 = float_to_q7dot8(OPUS_REF_R128_LOUDNESS_DBFS - album_loudness)
      assert(-32768 <= q78 <= 32767)
      mf["R128_ALBUM_GAIN"] = str(q78)

  elif (isinstance(mf.tags, (mutagen._vorbis.VComment, mutagen.apev2.APEv2)) or
          isinstance(mf, (mutagen.ogg.OggFileType, mutagen.apev2.APEv2File))):
    # https://wiki.xiph.org/VorbisComment#Replay_Gain
    if loudness is not None:
      mf["REPLAYGAIN_TRACK_GAIN"] = "%.2f dB" % (RG2_REF_R128_LOUDNESS_DBFS - loudness)
      mf["REPLAYGAIN_TRACK_PEAK"] = "%.8f" % (peak)
    if album_loudness is not None:
      mf["REPLAYGAIN_ALBUM_GAIN"] = "%.2f dB" % (RG2_REF_R128_LOUDNESS_DBFS - album_loudness)
      mf["REPLAYGAIN_ALBUM_PEAK"] = "%.8f" % (album_peak)

  elif (isinstance(mf.tags, mutagen.mp4.MP4Tags) or
        isinstance(mf, mutagen.mp4.MP4)):
    # https://github.com/xbmc/xbmc/blob/9e855967380ef3a5d25718ff2e6db5e3dd2e2829/xbmc/music/tags/TagLoaderTagLib.cpp#L806-L812
    if loudness is not None:
      mf["----:com.apple.iTunes:replaygain_track_gain"] = mutagen.mp4.MP4FreeForm(("%.2f dB" % (RG2_REF_R128_LOUDNESS_DBFS - loudness)).encode())
      mf["----:com.apple.iTunes:replaygain_track_peak"] = mutagen.mp4.MP4FreeForm(("%.6f" % (peak)).encode())
    if album_loudness is not None:
      mf["----:com.apple.iTunes:replaygain_album_gain"] = mutagen.mp4.MP4FreeForm(("%.2f dB" % (RG2_REF_R128_LOUDNESS_DBFS - album_loudness)).encode())
      mf["----:com.apple.iTunes:replaygain_album_peak"] = mutagen.mp4.MP4FreeForm(("%.6f" % (album_peak)).encode())

  else:
    raise RuntimeError("Unhandled '%s' tag format for file '%s'" % (mf.__class__.__name__,
                                                                    filepath))

  mf.save()

  # preserve original modification time, possibly increasing it by some seconds
  if mtime_second_offset is not None:
    if mtime_second_offset == 0:
      logger().debug("Restoring modification time for file %r" % (filepath))
    else:
      logger().debug("Restoring modification time for file '{}' (adding {} seconds)".format(filepath,
                                                                                            mtime_second_offset))
    os.utime(filepath, times=(os.stat(filepath).st_atime, original_mtime + mtime_second_offset))


def has_loudness_tag(filepath):
  """ Return a pair of booleans indicating if filepath has a RG or R128 track/album tag, or None if file is invalid. """
  track, album = False, False

  try:
    mf = mutagen.File(filepath)
  except mutagen.MutagenError as e:
    logger().warning("File %r %s: %s" % (filepath,
                                         e.__class__.__qualname__,
                                         e))
    return
  if mf is None:
    return

  if (isinstance(mf.tags, mutagen.id3.ID3) or
          isinstance(mf, mutagen.id3.ID3FileType)):
    track = ("TXXX:REPLAYGAIN_TRACK_GAIN" in mf) and ("TXXX:REPLAYGAIN_TRACK_PEAK" in mf)
    album = ("TXXX:REPLAYGAIN_ALBUM_GAIN" in mf) and ("TXXX:REPLAYGAIN_ALBUM_PEAK" in mf)

  elif isinstance(mf, mutagen.oggopus.OggOpus):
    track = "R128_TRACK_GAIN" in mf
    album = "R128_ALBUM_GAIN" in mf

  elif (isinstance(mf.tags, (mutagen._vorbis.VComment, mutagen.apev2.APEv2)) or
        isinstance(mf, (mutagen.ogg.OggFileType, mutagen.apev2.APEv2File))):
    track = ("REPLAYGAIN_TRACK_GAIN" in mf) and ("REPLAYGAIN_TRACK_PEAK" in mf)
    album = ("REPLAYGAIN_ALBUM_GAIN" in mf) and ("REPLAYGAIN_ALBUM_PEAK" in mf)

  elif (isinstance(mf.tags, mutagen.mp4.MP4Tags) or
        isinstance(mf, mutagen.mp4.MP4)):
    track = ("----:com.apple.iTunes:replaygain_track_gain" in mf) and ("----:com.apple.iTunes:replaygain_track_peak" in mf)
    album = ("----:com.apple.iTunes:replaygain_album_gain" in mf) and ("----:com.apple.iTunes:replaygain_album_peak" in mf)

  else:
    logger().warning("Unhandled '%s' tag format for file %r" % (mf.__class__.__name__,
                                                                filepath))
    return

  return track, album


def show_scan_report(audio_filepaths, album_dir, r128_data):
  """ Display loudness scan results. """
  # track loudness/peak
  for audio_filepath in audio_filepaths:
    try:
      loudness, peak = r128_data[audio_filepath]
    except KeyError:
      loudness, peak = "SKIPPED", "SKIPPED"
    else:
      loudness = "%.1f LUFS" % (loudness)
      if peak is None:
        peak = "-"
      else:
        peak = "%.1f dBFS" % (scale_to_gain(peak))
    logger().info("File %r: loudness = %s, sample peak = %s" % (audio_filepath, loudness, peak))

  # album loudness/peak
  if album_dir:
    try:
      album_loudness, album_peak = r128_data[ALBUM_GAIN_KEY]
    except KeyError:
      album_loudness, album_peak = "SKIPPED", "SKIPPED"
    else:
      album_loudness = "%.1f LUFS" % (album_loudness)
      if album_peak is None:
        album_peak = "-"
      else:
        album_peak = "%.1f dBFS" % (scale_to_gain(album_peak))
    logger().info("Album %r: loudness = %s, sample peak = %s" % (album_dir, album_loudness, album_peak))


def process(audio_filepaths, *, album_gain=False, opus_output_gain=False, mtime_second_offset=None, skip_tagged=False,
            thread_count=None, ffmpeg_path=None, dry_run=False, report=False):
  """ Analyze and tag input audio files. """
  error_count = 0

  with dynamic_tqdm(total=len(audio_filepaths) + int(album_gain),
                    desc="Analyzing audio loudness",
                    unit=" files",
                    leave=False) as progress:
    # analyze files
    r128_data = scan(audio_filepaths,
                     album_gain=album_gain,
                     skip_tagged=skip_tagged,
                     thread_count=thread_count,
                     ffmpeg_path=ffmpeg_path,
                     progress=progress,
                     boxed_error_count=[error_count])

  if report:
    show_scan_report(audio_filepaths,
                     os.path.dirname(audio_filepaths[0]) if album_gain else None,
                     r128_data)

  if dry_run:
    return error_count

  # tag
  try:
    album_loudness, album_peak = r128_data[ALBUM_GAIN_KEY]
  except KeyError:
    album_loudness, album_peak = None, None
  for audio_filepath in audio_filepaths:
    try:
      loudness, peak = r128_data[audio_filepath]
    except KeyError:
      if album_loudness is None:
        # file was skipped
        continue
      else:
        loudness, peak = None, None
    try:
      tag(audio_filepath, loudness, peak,
          album_loudness=album_loudness, album_peak=album_peak,
          opus_output_gain=opus_output_gain, mtime_second_offset=mtime_second_offset)
    except Exception as e:
      logger().error("Failed to tag file %r: %s %s" % (audio_filepath,
                                                       e.__class__.__qualname__,
                                                       e))
      error_count += 1

  return error_count


def process_recursive(directories, *, album_gain=False, opus_output_gain=False, mtime_second_offset=None,
                      skip_tagged=False, thread_count=None, ffmpeg_path=None, dry_run=False, report=False):
  """ Analyze and tag all audio files recursively found in input directories. """
  error_count = 0

  # walk directories
  albums_filepaths = []
  walk_stats = collections.OrderedDict(((k, 0) for k in("files", "dirs")))
  with dynamic_tqdm(desc="Analyzing directories",
                    unit=" dir",
                    postfix=walk_stats,
                    leave=False) as progress:
    for input_directory in directories:
      for root_dir, subdirs, filepaths in os.walk(input_directory, followlinks=False):
        audio_filepaths = tuple(map(functools.partial(os.path.join, root_dir),
                                    filter(is_audio_filepath,
                                           filepaths)))
        if audio_filepaths:
          albums_filepaths.append(audio_filepaths)

        if progress is not None:
          walk_stats["files"] += len(filepaths)
          walk_stats["dirs"] += 1
          progress.set_postfix(walk_stats, refresh=False)
          progress.update(1)

  # get optimal thread count
  if thread_count is None:
    thread_count = OPTIMAL_THREAD_COUNT

  executor = concurrent.futures.ThreadPoolExecutor(max_workers=thread_count)
  start_evt = threading.Event()
  futures = {}

  with dynamic_tqdm(total=len(albums_filepaths),
                    desc="Building work queue",
                    unit=" albums",
                    leave=False) as progress:
    # analysis futures
    for album_filepaths in albums_filepaths:
      dir_futures = scan(album_filepaths,
                         album_gain=album_gain,
                         skip_tagged=skip_tagged,
                         ffmpeg_path=ffmpeg_path,
                         executor=executor,
                         start_evt=start_evt)
      dir_futures = {k: (tuple(f for f in dir_futures.keys() if f is not k), v) for k, v in dir_futures.items()}
      futures.update(dir_futures)

      if progress is not None:
        progress.update(1)

  with dynamic_tqdm(total=sum(map(len, albums_filepaths)) + int(album_gain) * len(albums_filepaths),
                    desc="Analyzing audio loudness",
                    unit=" files",
                    leave=False,
                    smoothing=0) as progress:
    # get results
    start_evt.set()
    pending_futures = futures
    retained_futures = set()

    while futures:
      done_futures, pending_futures = concurrent.futures.wait(pending_futures,
                                                              return_when=concurrent.futures.FIRST_COMPLETED)

      for done_future in done_futures:
        other_dir_futures, _ = futures[done_future]

        if progress is not None:
          # update progress
          progress.update(1)

        # only tag when the whole directory is scanned
        if any(f not in retained_futures for f in other_dir_futures):
          retained_futures.add(done_future)
          continue

        # get album filepaths
        dir_futures = (done_future,) + other_dir_futures
        audio_filepaths = tuple(futures[f][1] for f in dir_futures if futures[f][1] != ALBUM_GAIN_KEY)

        # get analysis results for this directory
        r128_data = {}
        for dir_future in dir_futures:
          key = futures[dir_future][1]
          try:
            result = dir_future.result()
          except Exception as e:
            if album_gain and (key == ALBUM_GAIN_KEY):
              logger().warning("Failed to analyze files %s: %s %s" % (", ".join(repr(audio_filepath) for audio_filepath in audio_filepaths),
                                                                      e.__class__.__qualname__,
                                                                      e))
            else:
              logger().warning("Failed to analyze file %r: %s %s" % (key,
                                                                     e.__class__.__qualname__,
                                                                     e))
            error_count += 1
          else:
            if result is not None:
              r128_data[key] = result

        if report and audio_filepaths:
          show_scan_report(audio_filepaths,
                           os.path.dirname(audio_filepaths[0]) if album_gain else None,
                           r128_data)

        if not dry_run:
          # tag
          try:
            album_loudness, album_peak = r128_data[ALBUM_GAIN_KEY]
          except KeyError:
            album_loudness, album_peak = None, None
          for audio_filepath in audio_filepaths:
            try:
              loudness, peak = r128_data[audio_filepath]
            except KeyError:
              if album_loudness is None:
                # file was skipped
                continue
              else:
                loudness, peak = None, None
            try:
              tag(audio_filepath, loudness, peak,
                  album_loudness=album_loudness, album_peak=album_peak,
                  opus_output_gain=opus_output_gain,
                  mtime_second_offset=mtime_second_offset)
            except Exception as e:
              logger().error("Failed to tag file '%s': %s %s" % (audio_filepath,
                                                                 e.__class__.__qualname__,
                                                                 e))
              error_count += 1

        retained_futures -= set(other_dir_futures)
        for f in dir_futures:
          del futures[f]

  executor.shutdown(True)

  return error_count


def cl_main():
  # parse args
  arg_parser = argparse.ArgumentParser(description="r128gain v%s.%s" % (__version__, __doc__),
                                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  arg_parser.add_argument("path",
                          nargs="+",
                          help="Audio file paths, or directory paths for recursive mode")
  arg_parser.add_argument("-a",
                          "--album-gain",
                          action="store_true",
                          default=False,
                          help="Enable album gain")
  arg_parser.add_argument("-r",
                          "--recursive",
                          action="store_true",
                          default=False,
                          help="""Enable recursive mode: process audio files in directories and subdirectories.
                                  If album gain is enabled, all files in a directory are considered as part of the same
                                  album.""")
  arg_parser.add_argument("-s",
                          "--skip-tagged",
                          action="store_true",
                          default=False,
                          help="""Do not scan & tag files already having loudness tags.
                                  Warning: only enable if you are sure of the validity of the existing tags, as it can
                                  cause volume differences if existing tags are incorrect or coming from a old RGv1
                                  tagger.""")
  arg_parser.add_argument("-o",
                          "--opus-output-gain",
                          action="store_true",
                          default=False,
                          help="""For Opus files, write album or track gain in the 'output gain' Opus header (see
                                  https://tools.ietf.org/html/rfc7845#page-15). This gain is mandatory to apply for all
                                  Opus decoders so this should improve compatibility with players not supporting the
                                  R128 tags.
                                  Warning: This is EXPERIMENTAL, only use if you fully understand the implications.""")
  arg_parser.add_argument("-p",
                          "--preserve-times",
                          dest="mtime_second_offset",
                          nargs="?",
                          const=0,
                          action="store",
                          type=int,
                          default=None,
                          help="""Preserve modification times of tagged files,
                                  optionally adding MTIME_SECOND_OFFSET seconds. """)
  arg_parser.add_argument("-c",
                          "--thread-count",
                          type=int,
                          default=None,
                          help="Maximum number of tracks to scan in parallel. If not specified, autodetect CPU count")
  arg_parser.add_argument("-f",
                          "--ffmpeg-path",
                          default=shutil.which("ffmpeg"),
                          help="""Full file path of ffmpeg executable (only needed if not in PATH).
                                  If not specified, autodetect""")
  arg_parser.add_argument("-d",
                          "--dry-run",
                          action="store_true",
                          default=False,
                          help="Do not write any tags, only show scan results")
  arg_parser.add_argument("-v",
                          "--verbosity",
                          choices=("warning", "normal", "debug"),
                          default="normal",
                          dest="verbosity",
                          help="Level of logging output")
  args = arg_parser.parse_args()

  # setup logger
  logging_level = {"warning": logging.WARNING,
                   "normal": logging.INFO,
                   "debug": logging.DEBUG}
  logging.getLogger().setLevel(logging_level[args.verbosity])
  if logging_level[args.verbosity] == logging.DEBUG:
    fmt = "%(asctime)s %(threadName)s: %(message)s"
  else:
    fmt = "%(message)s"
  logging_formatter = colored_logging.ColoredFormatter(fmt=fmt)
  logging_handler = logging.StreamHandler()
  logging_handler.setFormatter(logging_formatter)
  logging.getLogger().addHandler(logging_handler)

  # show ffmpeg version
  ffmpeg_lib_versions = get_ffmpeg_lib_versions(args.ffmpeg_path)
  logger().debug("Detected FFmpeg lib versions: "
                 "%s" % (", ".join("%s: %u.%u.%u" % (lib_name,
                                                     (lib_version >> 24) & 0xff,
                                                     (lib_version >> 16) & 0xff,
                                                     (lib_version >> 8) & 0xff)
                                   for lib_name, lib_version in ffmpeg_lib_versions.items())))

  # main
  if args.recursive:
    err_count = process_recursive(args.path,
                                  album_gain=args.album_gain,
                                  opus_output_gain=args.opus_output_gain,
                                  mtime_second_offset=args.mtime_second_offset,
                                  skip_tagged=args.skip_tagged,
                                  thread_count=args.thread_count,
                                  ffmpeg_path=args.ffmpeg_path,
                                  dry_run=args.dry_run,
                                  report=logging.getLogger().isEnabledFor(logging.INFO) or args.dry_run)
  else:
    err_count = process(args.path,
                        album_gain=args.album_gain,
                        opus_output_gain=args.opus_output_gain,
                        mtime_second_offset=args.mtime_second_offset,
                        skip_tagged=args.skip_tagged,
                        thread_count=args.thread_count,
                        ffmpeg_path=args.ffmpeg_path,
                        dry_run=args.dry_run,
                        report=logging.getLogger().isEnabledFor(logging.INFO) or args.dry_run)

  if err_count > 0:
    exit(1)


if getattr(sys, "frozen", False):
  freeze_dir = os.path.dirname(sys.executable)
  os.environ["PATH"] = os.pathsep.join((freeze_dir, os.environ["PATH"]))


if __name__ == "__main__":
  cl_main()
