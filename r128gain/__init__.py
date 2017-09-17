#!/usr/bin/env python3

""" Scan audio files and tag them with ReplayGain/R128 loudness metadata. """

__version__ = "0.5.0"
__author__ = "desbma"
__license__ = "GPLv3"

import argparse
import concurrent.futures
import contextlib
import functools
import logging
import os
import shutil
import subprocess
import time

import mutagen

import r128gain.colored_logging as colored_logging


AUDIO_EXTENSIONS = frozenset(("flac", "ogg", "opus", "m4a", "mp3", "mpc", "wv"))
RG2_REF_R128_LOUDNESS_DBFS = -18
OPUS_REF_R128_LOUDNESS_DBFS = -23


def logger():
  return logging.getLogger("r128gain")


def is_audio_filepath(filepath):
  """ Return True if filepath is a supported audio file. """
  # TODO more robust way to identify audio files? (open with mutagen?)
  return os.path.splitext(filepath)[-1].lstrip(".").lower() in AUDIO_EXTENSIONS


def get_r128_loudness(audio_filepaths, *, calc_peak=True, enable_ffmpeg_threading=True, ffmpeg_path=None):
  """ Get R128 loudness loudness level and peak, in dbFS. """
  logger().info("Analyzing loudness of file%s %s" % ("s" if (len(audio_filepaths) > 1) else "",
                                                     ", ".join("'%s'" % (audio_filepath) for audio_filepath in audio_filepaths)))
  cmd = [ffmpeg_path or "ffmpeg",
         "-hide_banner", "-nostats"]
  for i, audio_filepath in enumerate(audio_filepaths):
    if not enable_ffmpeg_threading:
      cmd.extend(("-threads:%u" % (i), "1"))  # single decoding thread
    cmd.extend(("-i", audio_filepath))
  # TODO do for FFmpeg >= 3.3 only
  # if not enable_ffmpeg_threading:
  #   cmd.extend(("-filter_threads", "1"))  # single filter thread
  filter_params = {"framelog": "verbose"}
  if calc_peak:
    filter_params["peak"] = "true"
  cmd.extend(("-map", "a"))
  if len(audio_filepaths) > 1:
    cmd.extend(("-filter_complex",
                "%s; "
                "%sconcat=n=%u:v=0:a=1[ac]; "
                "[ac]ebur128=%s" % ("; ".join(("[%u:a]aformat=sample_rates=48000:channel_layouts=stereo[a%u]" % (i, i)) for i in range(len(audio_filepaths))),
                                    "".join(("[a%u]" % (i)) for i in range(len(audio_filepaths))),
                                    len(audio_filepaths),
                                    ":".join("%s=%s" % (k, v) for k, v in filter_params.items()))))

  else:
    cmd.extend(("-filter:a", "ebur128=%s" % (":".join("%s=%s" % (k, v) for k, v in filter_params.items()))))
  cmd.extend(("-f", "null", os.devnull))
  logger().debug(subprocess.list2cmdline(cmd))
  output = subprocess.check_output(cmd,
                                   stdin=subprocess.DEVNULL,
                                   stderr=subprocess.STDOUT,
                                   universal_newlines=True)
  output = output.splitlines()
  for i in reversed(range(len(output))):
    line = output[i]
    if line.startswith("[Parsed_ebur128") and line.endswith("Summary:"):
      break
  output = filter(None, map(str.strip, output[i:]))
  r128_stats = dict(tuple(map(str.strip, line.split(":", 1))) for line in output if not line.endswith(":"))
  r128_stats = {k: float(v.split(" ", 1)[0]) for k, v in r128_stats.items()}
  return r128_stats["I"], r128_stats.get("Peak")


def scan(audio_filepaths, *, album_gain=False, thread_count=None, ffmpeg_path=None, executor=None):
  """ Analyze files, and return a dictionary of filepath to loudness metadata or filepath to future if executor is not None. """
  r128_data = {}

  with contextlib.ExitStack() as cm:
    if executor is None:
      if thread_count is None:
        thread_count = len(os.sched_getaffinity(0))
      enable_ffmpeg_threading = thread_count > (len(audio_filepaths) + int(album_gain))
      executor = cm.enter_context(concurrent.futures.ThreadPoolExecutor(max_workers=thread_count))
      async = False
    else:
      enable_ffmpeg_threading = False
      async = True

    futures = {}
    calc_album_peak = False
    for audio_filepath in audio_filepaths:
      if os.path.splitext(audio_filepath)[-1].lower() == ".opus":
        # http://www.rfcreader.com/#rfc7845_line1060
        calc_peak = False
      else:
        calc_peak = True
        calc_album_peak = True
      futures[audio_filepath] = executor.submit(get_r128_loudness,
                                                (audio_filepath,),
                                                calc_peak=calc_peak,
                                                enable_ffmpeg_threading=enable_ffmpeg_threading,
                                                ffmpeg_path=ffmpeg_path)
    if album_gain:
      futures[0] = executor.submit(get_r128_loudness,
                                   audio_filepaths,
                                   calc_peak=calc_album_peak,
                                   enable_ffmpeg_threading=enable_ffmpeg_threading,
                                   ffmpeg_path=ffmpeg_path)

    if async:
      return futures

    for audio_filepath in audio_filepaths:
      try:
        r128_data[audio_filepath] = futures[audio_filepath].result()
      except Exception as e:
        # raise
        logger().warning("Failed to analyze file '%s': %s %s" % (audio_filepath,
                                                                 e.__class__.__qualname__,
                                                                 e))
    if album_gain:
      try:
        r128_data[0] = futures[0].result()
      except Exception as e:
        # raise
        logger().warning("Failed to analyze files %s: %s %s" % (", ".join("'%s'" % (audio_filepath) for audio_filepath in audio_filepaths),
                                                                e.__class__.__qualname__,
                                                                e))
  return r128_data


def float_to_q7dot8(f):
  """ Encode float f to a fixed point Q7.8 integer. """
  # https://en.wikipedia.org/wiki/Q_(number_format)#Float_to_Q
  return int(round(f * (2 ** 8), 0))


def tag(filepath, loudness, peak, *,
        album_loudness=None, album_peak=None):
  """ Tag audio file with loudness metadata. """
  logger().info("Tagging file '%s'" % (filepath))
  mf = mutagen.File(filepath)

  if (isinstance(mf.tags, mutagen.id3.ID3) or
      isinstance(mf, mutagen.id3.ID3FileType)):
    # http://wiki.hydrogenaud.io/index.php?title=ReplayGain_2.0_specification#ID3v2
    mf.tags.add(mutagen.id3.TXXX(encoding=mutagen.id3.Encoding.LATIN1,
                                 desc="REPLAYGAIN_TRACK_GAIN",
                                 text="%.2f dB" % (RG2_REF_R128_LOUDNESS_DBFS - loudness)))
    mf.tags.add(mutagen.id3.TXXX(encoding=mutagen.id3.Encoding.LATIN1,
                                 desc="REPLAYGAIN_TRACK_PEAK",
                                 text="%.6f" % (10 ** (peak / 20))))
    if album_loudness is not None:
      mf.tags.add(mutagen.id3.TXXX(encoding=mutagen.id3.Encoding.LATIN1,
                                   desc="REPLAYGAIN_ALBUM_GAIN",
                                   text="%.2f dB" % (RG2_REF_R128_LOUDNESS_DBFS - album_loudness)))
      mf.tags.add(mutagen.id3.TXXX(encoding=mutagen.id3.Encoding.LATIN1,
                                   desc="REPLAYGAIN_ALBUM_PEAK",
                                   text="%.6f" % (10 ** (album_peak / 20))))
    # other legacy formats:
    # http://wiki.hydrogenaud.io/index.php?title=ReplayGain_legacy_metadata_formats#ID3v2_RGAD
    # http://wiki.hydrogenaud.io/index.php?title=ReplayGain_legacy_metadata_formats#ID3v2_RVA2

  elif isinstance(mf, mutagen.oggopus.OggOpus):
    # https://wiki.xiph.org/OggOpus#Comment_Header
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
    mf["REPLAYGAIN_TRACK_GAIN"] = "%.2f dB" % (RG2_REF_R128_LOUDNESS_DBFS - loudness)
    # peak_dbfs = 20 * log10(max_sample) <=> max_sample = 10^(peak_dbfs / 20)
    mf["REPLAYGAIN_TRACK_PEAK"] = "%.8f" % (10 ** (peak / 20))
    if album_loudness is not None:
      mf["REPLAYGAIN_ALBUM_GAIN"] = "%.2f dB" % (RG2_REF_R128_LOUDNESS_DBFS - album_loudness)
      mf["REPLAYGAIN_ALBUM_PEAK"] = "%.8f" % (10 ** (album_peak / 20))

  elif (isinstance(mf.tags, mutagen.mp4.MP4Tags) or
        isinstance(mf, mutagen.mp4.MP4)):
    # https://github.com/xbmc/xbmc/blob/9e855967380ef3a5d25718ff2e6db5e3dd2e2829/xbmc/music/tags/TagLoaderTagLib.cpp#L806-L812
    mf["----:COM.APPLE.ITUNES:REPLAYGAIN_TRACK_GAIN"] = mutagen.mp4.MP4FreeForm(("%.2f dB" % (RG2_REF_R128_LOUDNESS_DBFS - loudness)).encode())
    mf["----:COM.APPLE.ITUNES:REPLAYGAIN_TRACK_PEAK"] = mutagen.mp4.MP4FreeForm(("%.6f" % (10 ** (peak / 20))).encode())
    if album_loudness is not None:
      mf["----:COM.APPLE.ITUNES:REPLAYGAIN_ALBUM_GAIN"] = mutagen.mp4.MP4FreeForm(("%.2f dB" % (RG2_REF_R128_LOUDNESS_DBFS - album_loudness)).encode())
      mf["----:COM.APPLE.ITUNES:REPLAYGAIN_ALBUM_PEAK"] = mutagen.mp4.MP4FreeForm(("%.6f" % (10 ** (album_peak / 20))).encode())

  else:
    logger().warning("Unhandled '%s' tag format for file '%s'" % (mf.__class__.__name__,
                                                                  filepath))
    return

  mf.save()


def show_scan_report(audio_filepaths, r128_data):
  """ Display loudness scan results. """
  # track loudness/peak
  for audio_filepath in audio_filepaths:
    try:
      loudness, peak = r128_data[audio_filepath]
    except KeyError:
      loudness, peak = "ERR", "ERR"
    else:
      loudness = "%.1f dbFS" % (loudness)
      if peak is None:
        peak = "-"
      else:
        peak = "%.1f dbFS" % (peak)
    logger().info("File '%s': loudness = %s, peak = %s" % (audio_filepath, loudness, peak))

  # album loudness/peak
  try:
    album_loudness, album_peak = r128_data[0]
  except KeyError:
    pass
  else:
    album_loudness = "%.1f dbFS" % (album_loudness)
    if album_peak is None:
      album_peak = "-"
    else:
      album_peak = "%.1f dbFS" % (album_peak)
    album_dir = os.path.dirname(audio_filepaths[0])
    logger().info("Album '%s': loudness = %s, peak = %s" % (album_dir, album_loudness, album_peak))


def process(audio_filepaths, *, album_gain=False, thread_count=None, ffmpeg_path=None, dry_run=False, report=False):
  # analyze files
  r128_data = scan(audio_filepaths,
                   album_gain=album_gain,
                   thread_count=thread_count,
                   ffmpeg_path=ffmpeg_path)

  if report:
    show_scan_report(audio_filepaths, r128_data)

  if dry_run:
    return

  # tag
  if album_gain:
    album_loudness, album_peak = r128_data[0]
  else:
    album_loudness, album_peak = None, None
  for audio_filepath in audio_filepaths:
    try:
      loudness, peak = r128_data[audio_filepath]
    except KeyError:
      continue
    try:
      tag(audio_filepath, loudness, peak,
          album_loudness=album_loudness, album_peak=album_peak)
    except Exception as e:
      logger().error("Failed to tag file '%s': %s %s" % (audio_filepath,
                                                         e.__class__.__qualname__,
                                                         e))


def process_recursive(directories, *, album_gain=False, thread_count=None, ffmpeg_path=None, dry_run=False,
                      report=False):
  if thread_count is None:
    thread_count = len(os.sched_getaffinity(0))

  dir_futures = {}
  with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
    # walk directories, start analysis
    for input_directory in directories:
      for root_dir, subdirs, filepaths in os.walk(input_directory, followlinks=False):
        audio_filepaths = tuple(map(functools.partial(os.path.join, root_dir),
                                    filter(is_audio_filepath,
                                           filepaths)))
        if audio_filepaths:
          dir_futures[root_dir] = scan(audio_filepaths,
                                       album_gain=album_gain,
                                       ffmpeg_path=ffmpeg_path,
                                       executor=executor)

    # get results
    while dir_futures:
      to_del = None
      for directory, current_dir_futures in dir_futures.items():
        done, not_done = concurrent.futures.wait(current_dir_futures.values(),
                                                 timeout=0)
        if not not_done:
          # get analysis results for this directory
          r128_data = {}
          audio_filepaths = tuple(filter(lambda x: isinstance(x, str),
                                         current_dir_futures.keys()))
          for key, future in current_dir_futures.items():
            try:
              r128_data[key] = future.result()
            except Exception as e:
              if album_gain and (key == 0):
                logger().warning("Failed to analyze files %s: %s %s" % (", ".join("'%s'" % (audio_filepath) for audio_filepath in audio_filepaths),
                                                                        e.__class__.__qualname__,
                                                                        e))
              else:
                logger().warning("Failed to analyze file '%s': %s %s" % (key,
                                                                         e.__class__.__qualname__,
                                                                         e))

          if report:
            show_scan_report(audio_filepaths, r128_data)

          if not dry_run:
            # tag
            if album_gain:
              album_loudness, album_peak = r128_data[0]
            else:
              album_loudness, album_peak = None, None
            for audio_filepath in audio_filepaths:
              try:
                loudness, peak = r128_data[audio_filepath]
              except KeyError:
                continue
              try:
                tag(audio_filepath, loudness, peak,
                    album_loudness=album_loudness, album_peak=album_peak)
              except Exception as e:
                logger().error("Failed to tag file '%s': %s %s" % (audio_filepath,
                                                                   e.__class__.__qualname__,
                                                                   e))

          # we are done with this directory
          to_del = directory
          break

      if to_del is not None:
        del dir_futures[to_del]
      else:
        # be nice with CPU usage, the real CPU intensive work is done by the executor
        time.sleep(0.3)


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

  # main
  try:
    if args.recursive:
      process_recursive(args.path,
                        album_gain=args.album_gain,
                        thread_count=args.thread_count,
                        ffmpeg_path=args.ffmpeg_path,
                        dry_run=args.dry_run,
                        report=logging.getLogger().isEnabledFor(logging.INFO) or args.dry_run)
    else:
      process(args.path,
              album_gain=args.album_gain,
              thread_count=args.thread_count,
              ffmpeg_path=args.ffmpeg_path,
              dry_run=args.dry_run,
              report=logging.getLogger().isEnabledFor(logging.INFO) or args.dry_run)
  except RuntimeError as e:
    logging.getLogger().error(e)
    exit(1)


if __name__ == "__main__":
  cl_main()
