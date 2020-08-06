"""
This file exists to provide one common place for all compression methods used in
simplifying and serializing PySyft objects.
"""
import zlib
import lz4
from lz4 import (  # noqa: F401
    frame,
)  # needed as otherwise we will get: module 'lz4' has no attribute 'frame'

from syft.exceptions import CompressionNotFoundException

# COMPRESSION SCHEME INT CODES
NO_COMPRESSION = 40
LZ4 = 41
ZLIB = 42
scheme_to_bytes = {
    NO_COMPRESSION: NO_COMPRESSION.to_bytes(1, byteorder="big"),
    LZ4: LZ4.to_bytes(1, byteorder="big"),
    ZLIB: ZLIB.to_bytes(1, byteorder="big"),
}
default_compress_scheme = LZ4

## SECTION: chosen Compression Algorithm


def _apply_compress_scheme(decompressed_input_bin) -> tuple:
    """
    Apply the selected compression scheme.
    By default is used LZ4

    Args:
        decompressed_input_bin: the binary to be compressed
    """
    return scheme_to_compression[default_compress_scheme](decompressed_input_bin)


def apply_zlib_compression(uncompressed_input_bin) -> tuple:
    """
    Apply zlib compression to the input

    Args:
        decompressed_input_bin: the binary to be compressed

    Returns:
        a tuple (compressed_result, ZLIB)
    """

    return zlib.compress(uncompressed_input_bin), ZLIB


def apply_lz4_compression(decompressed_input_bin) -> tuple:
    """
    Apply LZ4 compression to the input

    Args:
        decompressed_input_bin: the binary to be compressed

    Returns:
        a tuple (compressed_result, LZ4)
    """
    return lz4.frame.compress(decompressed_input_bin), LZ4


def apply_no_compression(decompressed_input_bin) -> tuple:
    """
    No compression is applied to the input

    Args:
        decompressed_input_bin: the binary

    Returns:
        a tuple (the binary, NO_COMPRESSION)
    """

    return decompressed_input_bin, NO_COMPRESSION


scheme_to_compression = {
    NO_COMPRESSION: apply_no_compression,
    LZ4: apply_lz4_compression,
    ZLIB: apply_zlib_compression,
}


def _compress(decompressed_input_bin: bin) -> bin:
    """
    This function compresses a binary using the function _apply_compress_scheme
    if the input has been already compressed in some step, it will return it as it is

    Args:
        decompressed_input_bin (bin): binary to be compressed

    Returns:
        bin: a compressed binary

    """
    compress_stream, compress_scheme = _apply_compress_scheme(decompressed_input_bin)
    try:
        z = scheme_to_bytes[compress_scheme] + compress_stream
        return z
    except KeyError:
        raise CompressionNotFoundException(
            f"Compression scheme not found for compression code: {str(compress_scheme)}"
        )


def _decompress(binary: bin) -> bin:
    """
    This function decompresses a binary using the scheme defined in the first byte of the input

    Args:
        binary (bin): a compressed binary

    Returns:
        bin: decompressed binary

    """

    # check the 1-byte header to check the compression scheme used
    compress_scheme = binary[0]

    # remove the 1-byte header from the input stream
    binary = binary[1:]
    # 1)  Decompress or return the original stream
    if compress_scheme == LZ4:
        return lz4.frame.decompress(binary)
    elif compress_scheme == ZLIB:
        return zlib.decompress(binary)
    elif compress_scheme == NO_COMPRESSION:
        return binary
    else:
        raise CompressionNotFoundException(
            f"Compression scheme not found for compression code: {str(compress_scheme)}"
        )
