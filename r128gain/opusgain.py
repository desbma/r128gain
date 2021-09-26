""" OggOpus parsing code. """

import struct
from typing import BinaryIO

import crcmod

# https://tools.ietf.org/html/rfc3533#section-6
# http://www.onicos.com/staff/iz/formats/ogg.html
# https://en.wikipedia.org/wiki/Ogg#Page_structure
OGG_FIRST_PAGE_HEADER = struct.Struct("<4sBBQLLLB")
OGG_FIRST_PAGE_HEADER_CRC_OFFSET = 22
OGG_FIRST_PAGE_HEADER_CRC = struct.Struct("<L")

# https://tools.ietf.org/html/rfc7845#section-5.1
OGG_OPUS_ID_HEADER = struct.Struct("<8sBBHLhB")
OGG_OPUS_ID_HEADER_GAIN_OFFSET = 16
OGG_OPUS_ID_HEADER_GAIN = struct.Struct("<h")

ogg_page_crc = crcmod.mkCrcFun(0x104C11DB7, initCrc=0, rev=False)


def parse_oggopus_output_gain(file: BinaryIO) -> int:
    # noqa: D200
    """
    Parse an OggOpus file headers, read and return its output gain, and set file seek position to start of Opus header.
    """
    #
    # Ogg header
    #

    # check fields of Ogg page header
    chunk = file.read(OGG_FIRST_PAGE_HEADER.size)
    first_ogg_page = bytearray()
    first_ogg_page.extend(chunk)
    if len(chunk) < OGG_FIRST_PAGE_HEADER.size:
        raise ValueError(
            "Not enough bytes in Ogg page header: %u, expected at least %u" % (len(chunk), OGG_FIRST_PAGE_HEADER.size)
        )
    (
        capture_pattern,
        version,
        header_type,
        granule_position,
        bitstream_serial_number,
        page_sequence_number,
        crc_checksum,
        page_segments,
    ) = OGG_FIRST_PAGE_HEADER.unpack(chunk)
    if capture_pattern != b"OggS":
        raise ValueError(f"Invalid OGG capture pattern: {repr(capture_pattern)}, expected 'OggS'")
    if version != 0:
        raise ValueError("Invalid OGG version: %u, expected %u" % (version, 0))
    if header_type != 2:  # should be first page of stream
        raise ValueError("Invalid OGG page header type: %u, expected %u" % (header_type, 2))
    if page_sequence_number != 0:
        raise ValueError("Invalid OGG page sequence number: %u, expected %u" % (page_sequence_number, 0))
    segment_table_fmt = struct.Struct("<%uB" % (page_segments))
    chunk = file.read(segment_table_fmt.size)
    first_ogg_page.extend(chunk)
    if len(chunk) < segment_table_fmt.size:
        raise ValueError(
            "Not enough bytes for OGG segment table: %u, expected at least %u" % (len(chunk), segment_table_fmt.size)
        )
    segment_table = segment_table_fmt.unpack(chunk)

    # check crc of first page
    first_ogg_page_size = OGG_FIRST_PAGE_HEADER.size + segment_table_fmt.size + sum(segment_table)
    chunk = file.read(sum(segment_table))
    first_ogg_page.extend(chunk)
    if len(first_ogg_page) < first_ogg_page_size:
        raise ValueError(
            "Not enough bytes for first OGG page: %u, expected at least %u" % (len(first_ogg_page), first_ogg_page_size)
        )
    computed_crc = _compute_ogg_page_crc(first_ogg_page)
    if computed_crc != crc_checksum:
        raise ValueError(f"Invalid OGG page CRC: 0x{crc_checksum:08x}, expected 0x{computed_crc:08x}")

    #
    # Opus header
    #
    chunk = first_ogg_page[OGG_FIRST_PAGE_HEADER.size + segment_table_fmt.size :][: segment_table[0]]
    if len(chunk) < OGG_OPUS_ID_HEADER.size:
        raise ValueError(
            "Not enough bytes for Opus Identification header: %u, "
            "expected at least %u" % (len(chunk), OGG_OPUS_ID_HEADER.size)
        )
    magic, version, channel_count, preskip, input_samplerate, output_gain, mapping_family = OGG_OPUS_ID_HEADER.unpack(
        chunk[: OGG_OPUS_ID_HEADER.size]
    )
    if magic != b"OpusHead":
        raise ValueError(f"Invalid Opus magic number: {repr(magic)}, expected 'OpusHead'")
    if (version >> 4) != 0:
        raise ValueError(f"Invalid Opus version: 0x{version:x}, expected 0x0-0xf")

    # seek to Opus header
    file.seek(OGG_FIRST_PAGE_HEADER.size + segment_table_fmt.size)

    return output_gain


def write_oggopus_output_gain(file: BinaryIO, new_output_gain: int) -> None:
    """
    Write output gain Opus header for a file.

    file must be an object successfully used by parse_oggopus_output_gain.
    """
    opus_header_pos = file.tell()
    assert opus_header_pos > 0

    # write Opus header with new gain
    file.seek(opus_header_pos + OGG_OPUS_ID_HEADER_GAIN_OFFSET)
    file.write(OGG_OPUS_ID_HEADER_GAIN.pack(new_output_gain))

    # compute page crc
    file.seek(0)
    page = file.read(opus_header_pos + OGG_OPUS_ID_HEADER.size)
    computed_crc = _compute_ogg_page_crc(page)

    # write CRC
    file.seek(OGG_FIRST_PAGE_HEADER_CRC_OFFSET)
    file.write(OGG_FIRST_PAGE_HEADER_CRC.pack(computed_crc))


def _compute_ogg_page_crc(page: bytes) -> int:
    """Compute CRC of an Ogg page."""
    page_zero_crc = (
        page[:OGG_FIRST_PAGE_HEADER_CRC_OFFSET]
        + b"\00" * OGG_FIRST_PAGE_HEADER_CRC.size
        + page[OGG_FIRST_PAGE_HEADER_CRC_OFFSET + OGG_FIRST_PAGE_HEADER_CRC.size :]
    )
    return ogg_page_crc(page_zero_crc)
