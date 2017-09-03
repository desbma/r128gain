#!/usr/bin/env python3

""" Scan audio files and tag them with ReplayGain/R128 loudness metadata. """

__version__ = "0.2.1"
__author__ = "desbma"
__license__ = "GPLv3"

import argparse
import concurrent.futures
import logging
import os
import shutil
import subprocess

import mutagen

import r128gain.colored_logging as colored_logging


def logger():
  return logging.getLogger("r128gain")


def get_r128_loudness(audio_filepath, *, ffmpeg_path=None, calc_peak=True, enable_ffmpeg_threading=True):
  """ Get R128 loudness level and peak, in dbFS. """
  logger().info("Analyzing loudness of file '%s'" % (audio_filepath))
  cmd = [ffmpeg_path or "ffmpeg",
         "-hide_banner", "-nostats"]
  if not enable_ffmpeg_threading:
    cmd.extend(("-threads", "1"))  # single decoding thread
  cmd.extend(("-i", audio_filepath,
              "-map", "a"))
  filter_params = {"framelog": "verbose"}
  if calc_peak:
    filter_params["peak"] = "true"
  if not enable_ffmpeg_threading:
    cmd.extend(("-filter_threads", "1"))  # single filter thread
  cmd.extend(("-filter:a", "ebur128=%s" % (":".join("%s=%s" % (k, v) for k, v in filter_params.items())),
              "-f", "null", os.devnull))
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


def scan(audio_filepaths, *, ffmpeg_path=None, thread_count=None):
  """ Analyze files, and return a dictionary of filepath to loudness metadata. """
  r128_data = {}

  if thread_count is None:
    thread_count = len(os.sched_getaffinity(0))
  enable_ffmpeg_threading = thread_count > len(audio_filepaths)

  with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
    futures = {}
    for audio_filepath in audio_filepaths:
      if os.path.splitext(audio_filepath)[-1].lower() == ".opus":
        # http://www.rfcreader.com/#rfc7845_line1060
        calc_peak = False
      else:
        calc_peak = True
      futures[audio_filepath] = executor.submit(get_r128_loudness,
                                                audio_filepath,
                                                ffmpeg_path=ffmpeg_path,
                                                calc_peak=calc_peak,
                                                enable_ffmpeg_threading=enable_ffmpeg_threading)

    for audio_filepath in audio_filepaths:
      try:
        r128_data[audio_filepath] = futures[audio_filepath].result()
      except Exception as e:
        # raise
        logger().warning("Failed to analyze file '%s': %s %e" % (audio_filepath,
                                                                 e.__class__.__qualname__,
                                                                 e))
  return r128_data


def float_to_q7dot8(f):
  """ Encode float f to a fixed point Q7.8 integer. """
  # https://en.wikipedia.org/wiki/Q_(number_format)#Float_to_Q
  return int(round(f * (2 ** 8), 0))


def tag(filepath, loudness, peak, ref_loudness):
  """ Tag audio file with loudness metadata. """
  logger().info("Tagging file '%s'" % (filepath))
  mf = mutagen.File(filepath)

  if isinstance(mf, mutagen.oggvorbis.OggVorbis):
    # https://wiki.xiph.org/VorbisComment#Replay_Gain
    mf["REPLAYGAIN_TRACK_GAIN"] = "%.2f dB" % (ref_loudness - loudness)
    # peak_dbfs = 20 * log10(max_sample) <=> max_sample = 10^(peak_dbfs / 20)
    mf["REPLAYGAIN_TRACK_PEAK"] = "%.8f" % (10 ** (peak / 20))

  elif isinstance(mf, mutagen.oggopus.OggOpus):
    # https://wiki.xiph.org/OggOpus#Comment_Header
    q78 = float_to_q7dot8(ref_loudness - loudness)
    assert(-32768 <= q78 <= 32767)
    mf["R128_TRACK_GAIN"] = str(q78)

  elif isinstance(mf, mutagen.mp3.MP3):
    # http://wiki.hydrogenaud.io/index.php?title=ReplayGain_2.0_specification#ID3v2
    mf.tags.add(mutagen.id3.TXXX(encoding=mutagen.id3.Encoding.LATIN1,
                                 desc="REPLAYGAIN_TRACK_GAIN",
                                 text="%.2f dB" % (ref_loudness - loudness)))
    mf.tags.add(mutagen.id3.TXXX(encoding=mutagen.id3.Encoding.LATIN1,
                                 desc="REPLAYGAIN_TRACK_PEAK",
                                 text="%.6f" % (10 ** (peak / 20))))
    # other legacy formats:
    # http://wiki.hydrogenaud.io/index.php?title=ReplayGain_legacy_metadata_formats#ID3v2_RGAD
    # http://wiki.hydrogenaud.io/index.php?title=ReplayGain_legacy_metadata_formats#ID3v2_RVA2

  elif isinstance(mf, mutagen.mp4.MP4):
    # https://github.com/xbmc/xbmc/blob/9e855967380ef3a5d25718ff2e6db5e3dd2e2829/xbmc/music/tags/TagLoaderTagLib.cpp#L806-L812
    mf["----:COM.APPLE.ITUNES:REPLAYGAIN_TRACK_GAIN"] = mutagen.mp4.MP4FreeForm(("%.2f dB" % (ref_loudness - loudness)).encode())
    mf["----:COM.APPLE.ITUNES:REPLAYGAIN_TRACK_PEAK"] = mutagen.mp4.MP4FreeForm(("%.6f" % (10 ** (peak / 20))).encode())

  mf.save()


def process(audio_filepaths, *, ref_loudness=-18, ffmpeg_path=None, thread_count=None, dry_run=False, report=False):
  # analyze files
  r128_data = scan(audio_filepaths,
                   ffmpeg_path=ffmpeg_path,
                   thread_count=thread_count)

  if report:
    # report
    max_len = max(map(len, audio_filepaths))
    cols = ("Filepath", "Level (dBFS)", "Peak (dBFS)")
    print(cols[0].ljust(max_len), cols[1], cols[2], sep="  ")
    for audio_filepath in audio_filepaths:
      try:
        level, peak = r128_data[audio_filepath]
      except KeyError:
        level, peak = "ERR", "ERR"
      else:
        if peak is None:
          peak = "-"
        level, peak = map(str, (level, peak))
      print(audio_filepath.ljust(max_len), level.ljust(len(cols[1])), peak, sep="  ")

  if dry_run:
    return

  # tag
  for audio_filepath in audio_filepaths:
    try:
      level, peak = r128_data[audio_filepath]
    except KeyError:
      continue
    try:
      tag(audio_filepath, level, peak, ref_loudness)
    except Exception as e:
      logger().error("Failed to tag file '%s': %s %s" % (audio_filepath,
                                                         e.__class__.__qualname__,
                                                         e))


def cl_main():
  # parse args
  arg_parser = argparse.ArgumentParser(description="r128gain v%s.%s" % (__version__, __doc__),
                                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  arg_parser.add_argument("filepath",
                          nargs="+",
                          help="Audio files")
  arg_parser.add_argument("-r",
                          "--reference-loudness",
                          type=int,
                          default=-18,
                          help="Reference loudness level in dBFS. Do not change unless you know what you are doing")
  arg_parser.add_argument("-c",
                          "--thread-count",
                          type=int,
                          default=None,
                          help="Maximum number of tracks to scan in parallel. If not specified, autodetect")
  arg_parser.add_argument("-f",
                          "--ffmpeg-path",
                          default=shutil.which("ffmpeg"),
                          help="Full file path of ffmpeg executable if not in PATH")
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
    process(args.filepath,
            ffmpeg_path=args.ffmpeg_path,
            thread_count=args.thread_count,
            dry_run=args.dry_run,
            ref_loudness=args.reference_loudness,
            report=logging.getLogger().isEnabledFor(logging.INFO) or args.dry_run)
  except RuntimeError as e:
    logging.getLogger().error(e)
    exit(1)


if __name__ == "__main__":
  cl_main()
