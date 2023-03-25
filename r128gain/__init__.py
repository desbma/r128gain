#!/usr/bin/env python3

""" Scan audio files and tag them with ReplayGain/R128 loudness metadata. """

__version__ = "1.0.7"
__author__ = "desbma"
__license__ = "LGPLv2"

import argparse
import collections
import concurrent.futures
import contextlib
import functools
import logging
import math
import mimetypes
import operator
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import ffmpeg
import mutagen
import tqdm

import r128gain.colored_logging as colored_logging
import r128gain.opusgain as opusgain
import r128gain.tqdm_logging as tqdm_logging

try:
    # Python >= 3.8
    cmd_to_string: Callable[[Sequence[str]], str] = shlex.join
except AttributeError:
    cmd_to_string = subprocess.list2cmdline


AUDIO_EXTENSIONS = frozenset(
    ("aac", "ape", "flac", "m4a", "mp3", "mp4", "mpc", "ogg", "oga", "opus", "tta", "wv")
) | frozenset(
    ext for ext, mime in {**mimetypes.types_map, **mimetypes.common_types}.items() if mime.startswith("audio/")
)
RG2_REF_R128_LOUDNESS_DBFS = -18
OPUS_REF_R128_LOUDNESS_DBFS = -23
ALBUM_GAIN_KEY = 0
try:
    OPTIMAL_THREAD_COUNT = len(os.sched_getaffinity(0))
except AttributeError:
    OPTIMAL_THREAD_COUNT = os.cpu_count()  # type: ignore
RE_ANULLSINK_REPLACE_OUTPUT = (re.compile(r"]anullsink(\[s\d+\])"), "]anullsink")


def logger() -> logging.Logger:
    """Get default logger."""
    return logging.getLogger("r128gain")


@contextlib.contextmanager
def dynamic_tqdm(*tqdm_args, **tqdm_kwargs):
    """Context manager that returns a tqdm object or None depending on context."""
    with contextlib.ExitStack() as cm:
        if sys.stderr.isatty() and logger().isEnabledFor(logging.INFO):
            progress = cm.enter_context(tqdm.tqdm(*tqdm_args, **tqdm_kwargs))
            cm.enter_context(tqdm_logging.redirect_logging(progress))
        else:
            progress = None
        yield progress


def is_audio_filepath(filepath: str) -> bool:
    """Return True if filepath is a supported audio file."""
    # TODO more robust way to identify audio files? (open with mutagen?)
    return os.path.splitext(filepath)[-1].lstrip(".").lower() in AUDIO_EXTENSIONS


@functools.lru_cache()
def get_ffmpeg_lib_versions(ffmpeg_path: Optional[str] = None) -> Dict[str, int]:
    """
    Get FFmpeg library versions as 32 bit integers, with same format as sys.hexversion.

    Example: 0x3040100 for FFmpeg 3.4.1
    """
    r = collections.OrderedDict()
    cmd = (ffmpeg_path or "ffmpeg", "-version")
    output_str = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, universal_newlines=True).stdout
    output_lines = output_str.splitlines()
    lib_version_regex = re.compile(r"^\s*(lib[a-z]+)\s+([0-9]+).\s*([0-9]+).\s*([0-9]+)\s+")
    for line in output_lines:
        match = lib_version_regex.search(line)
        if match:
            lib_name, *lib_version = match.group(1, 2, 3, 4)
            int_lib_version = 0
            for i, d in enumerate(map(int, reversed(lib_version)), 1):
                int_lib_version |= d << (8 * i)
            r[lib_name] = int_lib_version
    return r


def run_ffmpeg(cmd: List[str]) -> bytes:
    """Run FFmpeg command, and get stderr bytes."""
    # workaround https://github.com/kkroening/ffmpeg-python/issues/161 + python-ffmpeg incorrect handling of anullsink
    filter_opt_index = cmd.index("-filter_complex")
    filter_script = cmd[filter_opt_index + 1]
    anullsink_pads = []
    for match in RE_ANULLSINK_REPLACE_OUTPUT[0].finditer(filter_script):
        pad = match.group(1)
        anullsink_pads.append(pad)
    map_opt_idx = -1
    while True:
        try:
            map_opt_idx = cmd.index("-map", map_opt_idx + 1)
        except ValueError:
            break
        if cmd[map_opt_idx + 1] in anullsink_pads:
            cmd = cmd[:map_opt_idx] + cmd[map_opt_idx + 2 :]
            map_opt_idx -= 1
    filter_opt_index = cmd.index("-filter_complex")
    filter_script = RE_ANULLSINK_REPLACE_OUTPUT[0].sub(RE_ANULLSINK_REPLACE_OUTPUT[1], filter_script)
    logger().debug(f"Filter script: {filter_script}")
    with tempfile.TemporaryDirectory(prefix="r128gain_") as tmp_dir:
        tmp_script_filepath = os.path.join(tmp_dir, "ffmpeg_filters")
        with open(tmp_script_filepath, "wt") as f:
            f.write(filter_script)
        cmd[filter_opt_index] = "-filter_complex_script"
        cmd[filter_opt_index + 1] = tmp_script_filepath

        # run
        logger().debug(cmd_to_string(cmd))
        output = subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL, stderr=subprocess.PIPE).stderr
    return output


def get_r128_loudness(  # noqa: C901
    audio_filepaths: Sequence[str],
    *,
    calc_peak: bool = True,
    enable_ffmpeg_threading: bool = True,
    ffmpeg_path: Optional[str] = None,
    start_evt: Optional[threading.Event] = None,
) -> Tuple[float, Optional[float]]:
    """Get R128 loudness loudness level and sample peak."""
    if start_evt is not None:
        start_evt.wait()

    logger().info(
        "Analyzing loudness of file%s %s..."
        % (
            "s" if (len(audio_filepaths) > 1) else "",
            ", ".join(repr(audio_filepath) for audio_filepath in audio_filepaths),
        )
    )

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
        ffmpeg_input = ffmpeg_input.filter("aformat", sample_fmts="s16", sample_rates="48000", channel_layouts="stereo")
        if calc_peak:
            split_streams = ffmpeg_input.filter_multi_output("asplit", outputs=2)
            ffmpeg_rg_stream, ffmpeg_r128_stream = split_streams[0], split_streams[1]
            ffmpeg_rg_stream = ffmpeg_rg_stream.filter("replaygain")
            ffmpeg_rg_stream = ffmpeg_rg_stream.filter("anullsink")
            output_streams.append(ffmpeg_rg_stream)
        else:
            ffmpeg_r128_stream = ffmpeg_input
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
    cmd = ffmpeg.compile(
        ffmpeg.output(*output_streams, os.devnull, **additional_ffmpeg_args, f="null").global_args(
            "-hide_banner", "-nostats"
        ),
        cmd=ffmpeg_path or "ffmpeg",
    )

    # run
    output = run_ffmpeg(cmd)
    output_lines = output.decode("utf-8", errors="replace").splitlines()

    if calc_peak:
        # parse replaygain filter output
        sample_peaks = []
        for line in reversed(output_lines):
            if line.startswith("[Parsed_replaygain_") and ("] track_peak = " in line):
                sample_peaks.append(float(line.rsplit("=", 1)[1]))
                if len(sample_peaks) == len(audio_filepaths):
                    break
        sample_peak: Optional[float] = max(sample_peaks)
    else:
        sample_peak = None

    # parse r128 filter output
    unrelated_line = True
    for line in filter(str.strip, output_lines):
        if line.startswith("[Parsed_ebur128_") and line.endswith("Summary:"):
            output_lines_r128 = [line.strip()]
            unrelated_line = False
        elif not unrelated_line:
            if line.startswith(" "):
                output_lines_r128.append(line.strip())
            else:
                unrelated_line = True
    r128_stats_raw: Dict[str, str] = dict(
        tuple(map(str.strip, line.split(":", 1)))  # type: ignore
        for line in output_lines_r128
        if not line.endswith(":")
    )
    r128_stats: Dict[str, float] = {k: float(v.split(" ", 1)[0]) for k, v in r128_stats_raw.items()}

    return r128_stats["I"], sample_peak


def scan(
    audio_filepaths: Sequence[str],
    *,
    album_gain: bool = False,
    skip_tagged: bool = False,
    thread_count: Optional[int] = None,
    ffmpeg_path: Optional[str] = None,
    executor: Optional[concurrent.futures.Executor] = None,
    progress: Optional[tqdm.tqdm] = None,
    boxed_error_count: Optional[List[int]] = None,
    start_evt: Optional[threading.Event] = None,
) -> Union[Dict[Union[str, int], Tuple[float, Optional[float]]], Dict[concurrent.futures.Future, Union[str, int]]]:
    # noqa: D205,D400,D415
    """
    Analyze files, and return a dictionary of filepath to loudness metadata or future to filepath if executor is not
    None.
    """
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
        audio_filepaths = tuple(
            audio_filepath for (audio_filepath, has_tags) in zip(audio_filepaths, loudness_tags) if has_tags is not None
        )
        loudness_tags = tuple(filter(None, loudness_tags))

        futures: Dict[concurrent.futures.Future, Union[str, int]] = {}
        if album_gain:
            if skip_tagged and all(map(operator.itemgetter(1), loudness_tags)):
                logger().info("All files already have an album gain tag, skipping album gain scan")
            elif audio_filepaths:
                calc_album_peak = any(map(lambda x: os.path.splitext(x)[-1].lower() != ".opus", audio_filepaths))
                future = executor.submit(
                    get_r128_loudness,
                    audio_filepaths,
                    calc_peak=calc_album_peak,
                    enable_ffmpeg_threading=enable_ffmpeg_threading,
                    ffmpeg_path=ffmpeg_path,
                    start_evt=start_evt,
                )
                futures[future] = ALBUM_GAIN_KEY
        audio_filepath: Union[str, int]
        for audio_filepath, has_tags in zip(audio_filepaths, loudness_tags):
            assert has_tags is not None
            if skip_tagged and has_tags[0]:
                logger().info(f"File {audio_filepath!r} already has a track gain tag, skipping track gain scan")
                # create dummy future
                future = executor.submit(lambda: None)  # type: ignore
            else:
                if os.path.splitext(audio_filepath)[-1].lower() == ".opus":
                    # http://www.rfcreader.com/#rfc7845_line1060
                    calc_peak = False
                else:
                    calc_peak = True
                future = executor.submit(
                    get_r128_loudness,
                    (audio_filepath,),
                    calc_peak=calc_peak,
                    enable_ffmpeg_threading=enable_ffmpeg_threading,
                    ffmpeg_path=ffmpeg_path,
                    start_evt=start_evt,
                )
            futures[future] = audio_filepath

        if asynchronous:
            return futures

        while futures:
            done_futures, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)

            for done_future in done_futures:
                audio_filepath = futures[done_future]
                try:
                    result = done_future.result()
                except Exception as e:
                    if audio_filepath == ALBUM_GAIN_KEY:
                        logger().warning(
                            "Failed to analyze files %s: %s %s"
                            % (
                                ", ".join(repr(audio_filepath) for audio_filepath in audio_filepaths),
                                e.__class__.__qualname__,
                                e,
                            )
                        )
                    else:
                        logger().warning(f"Failed to analyze file {audio_filepath!r}: {e.__class__.__qualname__} {e}")
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


def float_to_q7dot8(f: float) -> int:
    """Encode float f to a fixed point Q7.8 integer."""
    # https://en.wikipedia.org/wiki/Q_(number_format)#Float_to_Q
    return int(round(f * (2**8), 0))


def gain_to_scale(gain: float) -> float:
    """Convert a gain value in dBFS to a float where 1.0 is 0 dBFS."""
    return 10 ** (gain / 20)


def scale_to_gain(scale: float) -> float:
    """Convert a float value to a gain where 0 dBFS is 1.0."""
    if scale == 0:
        return -math.inf
    return 20 * math.log10(scale)


def tag(  # noqa: C901
    filepath: str,
    loudness: Optional[float],
    peak: Optional[float],
    *,
    album_loudness: Optional[float] = None,
    album_peak: Optional[float] = None,
    opus_output_gain: bool = False,
    mtime_second_offset: Optional[int] = None,
) -> None:
    """Tag audio file with loudness metadata."""
    assert (loudness is not None) or (album_loudness is not None)

    if peak is not None:
        assert 0 <= peak <= 1.0
        if album_peak is not None:
            assert 0 <= album_peak <= 1.0

    logger().info(f"Tagging file {filepath!r}")
    original_mtime = os.path.getmtime(filepath)
    mf = mutagen.File(filepath)
    if (mf is not None) and (mf.tags is None) and isinstance(mf, (mutagen.trueaudio.TrueAudio, mutagen.aac.AAC)):
        # some formats can have ID3 or APE tags, try to use APE if we already have some APE tags and no ID3
        try:
            mf_ape = mutagen.apev2.APEv2File(filepath)
        except Exception:
            pass
        else:
            if mf_ape.tags is not None:
                mf = mf_ape
    if (mf is not None) and (mf.tags is None):
        if isinstance(mf, (mutagen.trueaudio.TrueAudio, mutagen.aac.AAC)):
            mf = mutagen.apev2.APEv2File(filepath)
        mf.add_tags()

    if isinstance(mf.tags, mutagen.id3.ID3):
        # http://wiki.hydrogenaud.io/index.php?title=ReplayGain_2.0_specification#ID3v2
        if loudness is not None:
            assert peak is not None
            mf.tags.add(
                mutagen.id3.TXXX(
                    encoding=mutagen.id3.Encoding.LATIN1,
                    desc="REPLAYGAIN_TRACK_GAIN",
                    text=f"{RG2_REF_R128_LOUDNESS_DBFS - loudness:.2f} dB",
                )
            )
            mf.tags.add(
                mutagen.id3.TXXX(encoding=mutagen.id3.Encoding.LATIN1, desc="REPLAYGAIN_TRACK_PEAK", text=f"{peak:.6f}")
            )
        if album_loudness is not None:
            assert album_peak is not None
            mf.tags.add(
                mutagen.id3.TXXX(
                    encoding=mutagen.id3.Encoding.LATIN1,
                    desc="REPLAYGAIN_ALBUM_GAIN",
                    text=f"{RG2_REF_R128_LOUDNESS_DBFS - album_loudness:.2f} dB",
                )
            )
            mf.tags.add(
                mutagen.id3.TXXX(
                    encoding=mutagen.id3.Encoding.LATIN1, desc="REPLAYGAIN_ALBUM_PEAK", text=f"{album_peak:.6f}"
                )
            )
        # other legacy formats:
        # http://wiki.hydrogenaud.io/index.php?title=ReplayGain_legacy_metadata_formats#ID3v2_RGAD
        # http://wiki.hydrogenaud.io/index.php?title=ReplayGain_legacy_metadata_formats#ID3v2_RVA2

    elif isinstance(mf, mutagen.oggopus.OggOpus):
        if opus_output_gain:
            with open(filepath, "r+b") as file:
                current_output_gain = opusgain.parse_oggopus_output_gain(file)
                if album_loudness is not None:
                    # write header relative to album level
                    assert album_loudness is not None
                    output_gain_loudness = album_loudness
                else:
                    assert loudness is not None
                    output_gain_loudness = loudness
                new_output_gain = current_output_gain + float_to_q7dot8(
                    OPUS_REF_R128_LOUDNESS_DBFS - output_gain_loudness
                )
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
            assert -32768 <= q78 <= 32767
            mf["R128_TRACK_GAIN"] = str(q78)
        if album_loudness is not None:
            q78 = float_to_q7dot8(OPUS_REF_R128_LOUDNESS_DBFS - album_loudness)
            assert -32768 <= q78 <= 32767
            mf["R128_ALBUM_GAIN"] = str(q78)

    elif isinstance(mf.tags, (mutagen._vorbis.VComment, mutagen.apev2.APEv2)):
        # https://wiki.xiph.org/VorbisComment#Replay_Gain
        if loudness is not None:
            assert peak is not None
            mf["REPLAYGAIN_TRACK_GAIN"] = f"{RG2_REF_R128_LOUDNESS_DBFS - loudness:.2f} dB"
            mf["REPLAYGAIN_TRACK_PEAK"] = f"{peak:.8f}"
        if album_loudness is not None:
            assert album_peak is not None
            mf["REPLAYGAIN_ALBUM_GAIN"] = f"{RG2_REF_R128_LOUDNESS_DBFS - album_loudness:.2f} dB"
            mf["REPLAYGAIN_ALBUM_PEAK"] = f"{album_peak:.8f}"

    elif isinstance(mf.tags, mutagen.mp4.MP4Tags):
        # https://github.com/xbmc/xbmc/blob/9e855967380ef3a5d25718ff2e6db5e3dd2e2829/xbmc/music/tags/TagLoaderTagLib.cpp#L806-L812
        if loudness is not None:
            assert peak is not None
            mf["----:com.apple.iTunes:replaygain_track_gain"] = mutagen.mp4.MP4FreeForm(
                f"{RG2_REF_R128_LOUDNESS_DBFS - loudness:.2f} dB".encode()
            )
            mf["----:com.apple.iTunes:replaygain_track_peak"] = mutagen.mp4.MP4FreeForm(f"{peak:.6f}".encode())
        if album_loudness is not None:
            assert album_peak is not None
            mf["----:com.apple.iTunes:replaygain_album_gain"] = mutagen.mp4.MP4FreeForm(
                f"{RG2_REF_R128_LOUDNESS_DBFS - album_loudness:.2f} dB".encode()
            )
            mf["----:com.apple.iTunes:replaygain_album_peak"] = mutagen.mp4.MP4FreeForm(f"{album_peak:.6f}".encode())

    else:
        raise RuntimeError(f"Unhandled {mf.__class__.__qualname__!r} tag format for file {filepath!r}")

    mf.save()

    # preserve original modification time, possibly increasing it by some seconds
    if mtime_second_offset is not None:
        if mtime_second_offset == 0:
            logger().debug(f"Restoring modification time for file {filepath!r}")
        else:
            logger().debug(f"Restoring modification time for file {filepath!r} (adding {mtime_second_offset} seconds)")
        os.utime(filepath, times=(os.stat(filepath).st_atime, original_mtime + mtime_second_offset))


def has_loudness_tag(filepath: str) -> Optional[Tuple[bool, bool]]:
    # noqa: D200
    """
    Return a pair of booleans indicating if filepath has a RG or R128 track/album tag, or None if file is invalid.
    """
    track, album = False, False

    try:
        mf = mutagen.File(filepath)
    except mutagen.MutagenError as e:
        logger().warning(f"File {filepath!r} {e.__class__.__qualname__}: {e}")
        return None
    if mf is None:
        return None

    if isinstance(mf.tags, mutagen.id3.ID3):
        track = ("TXXX:REPLAYGAIN_TRACK_GAIN" in mf) and ("TXXX:REPLAYGAIN_TRACK_PEAK" in mf)
        album = ("TXXX:REPLAYGAIN_ALBUM_GAIN" in mf) and ("TXXX:REPLAYGAIN_ALBUM_PEAK" in mf)

    elif isinstance(mf, mutagen.oggopus.OggOpus):
        track = "R128_TRACK_GAIN" in mf
        album = "R128_ALBUM_GAIN" in mf

    elif isinstance(mf.tags, (mutagen._vorbis.VComment, mutagen.apev2.APEv2)):
        track = ("REPLAYGAIN_TRACK_GAIN" in mf) and ("REPLAYGAIN_TRACK_PEAK" in mf)
        album = ("REPLAYGAIN_ALBUM_GAIN" in mf) and ("REPLAYGAIN_ALBUM_PEAK" in mf)

    elif isinstance(mf.tags, mutagen.mp4.MP4Tags):
        track = ("----:com.apple.iTunes:replaygain_track_gain" in mf) and (
            "----:com.apple.iTunes:replaygain_track_peak" in mf
        )
        album = ("----:com.apple.iTunes:replaygain_album_gain" in mf) and (
            "----:com.apple.iTunes:replaygain_album_peak" in mf
        )

    else:
        logger().warning(f"Unhandled or missing {mf.__class__.__qualname__!r} tag format for file {filepath!r}")

    return track, album


def show_scan_report(
    audio_filepaths: Sequence[str],
    album_dir: Optional[str],
    r128_data: Dict[Union[str, int], Tuple[float, Optional[float]]],
):
    """Display loudness scan results."""
    # track loudness/peak
    for audio_filepath in audio_filepaths:
        try:
            loudness, peak = r128_data[audio_filepath]
        except KeyError:
            loudness_str, peak_str = "SKIPPED", "SKIPPED"
        else:
            loudness_str = f"{loudness:.1f} LUFS"
            if peak is None:
                peak_str = "-"
            else:
                peak_str = f"{scale_to_gain(peak):.1f} dBFS"
        logger().info(f"File {audio_filepath!r}: loudness = {loudness_str}, sample peak = {peak_str}")

    # album loudness/peak
    if album_dir:
        try:
            album_loudness, album_peak = r128_data[ALBUM_GAIN_KEY]
        except KeyError:
            album_loudness_str, album_peak_str = "SKIPPED", "SKIPPED"
        else:
            album_loudness_str = f"{album_loudness:.1f} LUFS"
            if album_peak is None:
                album_peak_str = "-"
            else:
                album_peak_str = f"{scale_to_gain(album_peak):.1f} dBFS"
        logger().info(f"Album {album_dir!r}: loudness = {album_loudness_str}, sample peak = {album_peak_str}")


def process(
    audio_filepaths: Sequence[str],
    *,
    album_gain: bool = False,
    opus_output_gain: bool = False,
    mtime_second_offset: Optional[int] = None,
    skip_tagged: bool = False,
    thread_count: Optional[int] = None,
    ffmpeg_path: Optional[str] = None,
    dry_run: bool = False,
    report: bool = False,
):
    """Analyze and tag input audio files."""
    error_count = 0

    with dynamic_tqdm(
        total=len(audio_filepaths) + int(album_gain), desc="Analyzing audio loudness", unit=" files", leave=False
    ) as progress:
        # analyze files
        r128_data: Dict[Union[str, int], Tuple[float, Optional[float]]] = scan(  # type: ignore
            audio_filepaths,
            album_gain=album_gain,
            skip_tagged=skip_tagged,
            thread_count=thread_count,
            ffmpeg_path=ffmpeg_path,
            progress=progress,
            boxed_error_count=[error_count],
        )

    if report:
        show_scan_report(
            audio_filepaths,
            os.path.dirname(audio_filepaths[0]) if album_gain else None,
            r128_data,
        )

    if dry_run:
        return error_count

    # tag
    try:
        album_loudness: Optional[float]
        album_peak: Optional[float]
        album_loudness, album_peak = r128_data[ALBUM_GAIN_KEY]
    except KeyError:
        album_loudness, album_peak = None, None
    for audio_filepath in audio_filepaths:
        try:
            loudness: Optional[float]
            peak: Optional[float]
            loudness, peak = r128_data[audio_filepath]
        except KeyError:
            if album_loudness is None:
                # file was skipped
                continue
            else:
                loudness, peak = None, None
        try:
            tag(
                audio_filepath,
                loudness,
                peak,
                album_loudness=album_loudness,
                album_peak=album_peak,
                opus_output_gain=opus_output_gain,
                mtime_second_offset=mtime_second_offset,
            )
        except Exception as e:
            logger().error(f"Failed to tag file {audio_filepath!r}: {e.__class__.__qualname__} {e}")
            error_count += 1

    return error_count


def process_recursive(  # noqa: C901
    directories: Sequence[str],
    *,
    album_gain: bool = False,
    opus_output_gain: bool = False,
    mtime_second_offset: Optional[int] = None,
    skip_tagged: bool = False,
    thread_count: Optional[int] = None,
    ffmpeg_path: Optional[str] = None,
    dry_run: bool = False,
    report: bool = False,
):
    """Analyze and tag all audio files recursively found in input directories."""
    error_count = 0

    # walk directories
    albums_filepaths = []
    walk_stats = collections.OrderedDict((k, 0) for k in ("files", "dirs"))
    with dynamic_tqdm(desc="Analyzing directories", unit=" dir", postfix=walk_stats, leave=True) as progress:
        for input_directory in directories:
            if not os.path.isdir(input_directory):
                logging.getLogger().warning(f"{input_directory!r} is not a directory, ignoring")
                continue
            for root_dir, subdirs, filepaths in os.walk(input_directory, followlinks=False):
                audio_filepaths = tuple(
                    map(functools.partial(os.path.join, root_dir), filter(is_audio_filepath, filepaths))
                )
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
    futures: Dict[concurrent.futures.Future, Tuple[Tuple[concurrent.futures.Future, ...], str]] = {}

    with dynamic_tqdm(total=len(albums_filepaths), desc="Building work queue", unit=" albums", leave=False) as progress:
        # analysis futures
        for album_filepaths in albums_filepaths:
            new_futures = scan(
                album_filepaths,
                album_gain=album_gain,
                skip_tagged=skip_tagged,
                ffmpeg_path=ffmpeg_path,
                executor=executor,
                start_evt=start_evt,
            )
            new_dir_futures: Dict[concurrent.futures.Future, Tuple[Tuple[concurrent.futures.Future, ...], str]] = {
                k: (tuple(f for f in new_futures.keys() if f is not k), v)  # type: ignore
                for k, v in new_futures.items()
            }
            futures.update(new_dir_futures)

            if progress is not None:
                progress.update(1)

    with dynamic_tqdm(
        total=sum(map(len, albums_filepaths)) + int(album_gain) * len(albums_filepaths),
        desc="Analyzing audio loudness",
        unit=" files",
        leave=True,
        smoothing=0,
    ) as progress:
        # get results
        start_evt.set()
        pending_futures = set(futures)
        retained_futures = set()

        while futures:
            done_futures, pending_futures = concurrent.futures.wait(
                pending_futures, return_when=concurrent.futures.FIRST_COMPLETED
            )

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
                r128_data: Dict[Union[str, int], Tuple[float, Optional[float]]] = {}
                for dir_future in dir_futures:
                    key = futures[dir_future][1]
                    try:
                        result = dir_future.result()
                    except Exception as e:
                        if album_gain and (key == ALBUM_GAIN_KEY):
                            logger().warning(
                                "Failed to analyze files %s: %s %s"
                                % (
                                    ", ".join(repr(audio_filepath) for audio_filepath in audio_filepaths),
                                    e.__class__.__qualname__,
                                    e,
                                )
                            )
                        else:
                            logger().warning(f"Failed to analyze file {key!r}: {e.__class__.__qualname__} {e}")
                        error_count += 1
                    else:
                        if result is not None:
                            r128_data[key] = result

                if report and audio_filepaths:
                    show_scan_report(
                        audio_filepaths,
                        os.path.dirname(audio_filepaths[0]) if album_gain else None,
                        r128_data,
                    )

                if not dry_run:
                    # tag
                    try:
                        album_loudness: Optional[float]
                        album_peak: Optional[float]
                        album_loudness, album_peak = r128_data[ALBUM_GAIN_KEY]
                    except KeyError:
                        album_loudness, album_peak = None, None
                    for audio_filepath in audio_filepaths:
                        try:
                            loudness: Optional[float]
                            peak: Optional[float]
                            loudness, peak = r128_data[audio_filepath]
                        except KeyError:
                            if album_loudness is None:
                                # file was skipped
                                continue
                            else:
                                loudness, peak = None, None
                        try:
                            tag(
                                audio_filepath,
                                loudness,
                                peak,
                                album_loudness=album_loudness,
                                album_peak=album_peak,
                                opus_output_gain=opus_output_gain,
                                mtime_second_offset=mtime_second_offset,
                            )
                        except Exception as e:
                            logger().error(f"Failed to tag file {audio_filepath!r}: {e.__class__.__qualname__} {e}")
                            error_count += 1

                retained_futures -= set(other_dir_futures)
                for f in dir_futures:
                    del futures[f]

    executor.shutdown(True)

    return error_count


def cl_main() -> None:
    """Command line entry point."""
    # parse args
    arg_parser = argparse.ArgumentParser(
        description=f"r128gain v{__version__}.{__doc__}", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    arg_parser.add_argument("path", nargs="+", help="Audio file paths, or directory paths for recursive mode")
    arg_parser.add_argument("-a", "--album-gain", action="store_true", default=False, help="Enable album gain")
    arg_parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        default=False,
        help="""Enable recursive mode: process audio files in directories and subdirectories.
                If album gain is enabled, all files in a directory are considered as part of the same
                album.""",
    )
    arg_parser.add_argument(
        "-s",
        "--skip-tagged",
        action="store_true",
        default=False,
        help="""Do not scan & tag files already having loudness tags.
                Warning: only enable if you are sure of the validity of the existing tags, as it can
                cause volume differences if existing tags are incorrect or coming from a old RGv1
                tagger.""",
    )
    arg_parser.add_argument(
        "-o",
        "--opus-output-gain",
        action="store_true",
        default=False,
        help="""For Opus files, write album or track gain in the 'output gain' Opus header (see
                https://tools.ietf.org/html/rfc7845#page-15). This gain is mandatory to apply for all
                Opus decoders so this should improve compatibility with players not supporting the
                R128 tags.
                Warning: This is EXPERIMENTAL, only use if you fully understand the implications.""",
    )
    arg_parser.add_argument(
        "-p",
        "--preserve-times",
        dest="mtime_second_offset",
        nargs="?",
        const=0,
        action="store",
        type=int,
        default=None,
        help="""Preserve modification times of tagged files,
                optionally adding MTIME_SECOND_OFFSET seconds. """,
    )
    arg_parser.add_argument(
        "-c",
        "--thread-count",
        type=int,
        default=None,
        help="Maximum number of tracks to scan in parallel. If not specified, autodetect CPU count",
    )
    arg_parser.add_argument(
        "-f",
        "--ffmpeg-path",
        default=shutil.which("ffmpeg"),
        help="""Full file path of ffmpeg executable (only needed if not in PATH).
                If not specified, autodetect""",
    )
    arg_parser.add_argument(
        "-d", "--dry-run", action="store_true", default=False, help="Do not write any tags, only show scan results"
    )
    arg_parser.add_argument(
        "-v",
        "--verbosity",
        choices=("warning", "normal", "debug"),
        default="normal",
        dest="verbosity",
        help="Level of logging output",
    )
    args = arg_parser.parse_args()

    # setup logger
    logging_level = {"warning": logging.WARNING, "normal": logging.INFO, "debug": logging.DEBUG}
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
    logger().debug(
        "Detected FFmpeg lib versions: "
        "%s"
        % (
            ", ".join(
                "%s: %u.%u.%u"
                % (lib_name, (lib_version >> 24) & 0xFF, (lib_version >> 16) & 0xFF, (lib_version >> 8) & 0xFF)
                for lib_name, lib_version in ffmpeg_lib_versions.items()
            )
        )
    )

    # main
    if args.recursive:
        err_count = process_recursive(
            args.path,
            album_gain=args.album_gain,
            opus_output_gain=args.opus_output_gain,
            mtime_second_offset=args.mtime_second_offset,
            skip_tagged=args.skip_tagged,
            thread_count=args.thread_count,
            ffmpeg_path=args.ffmpeg_path,
            dry_run=args.dry_run,
            report=logging.getLogger().isEnabledFor(logging.INFO) or args.dry_run,
        )
    else:
        err_count = process(
            args.path,
            album_gain=args.album_gain,
            opus_output_gain=args.opus_output_gain,
            mtime_second_offset=args.mtime_second_offset,
            skip_tagged=args.skip_tagged,
            thread_count=args.thread_count,
            ffmpeg_path=args.ffmpeg_path,
            dry_run=args.dry_run,
            report=logging.getLogger().isEnabledFor(logging.INFO) or args.dry_run,
        )

    if err_count > 0:
        # freeze target can not use exit directly
        sys.exit(1)


if getattr(sys, "frozen", False):
    freeze_dir = os.path.dirname(sys.executable)
    os.environ["PATH"] = os.pathsep.join((freeze_dir, os.environ["PATH"]))


if __name__ == "__main__":
    cl_main()
