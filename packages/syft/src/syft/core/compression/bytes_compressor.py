import lzma
import blosc

from .compression_flags import compression_params

class BytesCompressor:
    """
    [Experimental: high-performace duet channel] base class for specialized compression algorithms
    """
    libmap = {
        'lzma': lzma,
        'blosc': blosc
    }
    default_lib = 'lzma'
    default_cname = 'zlib'
    default_compression_lvl = 8

    @staticmethod
    def compress(blob: bytes) -> bytes:
        compression_lib = BytesCompressor.libmap[compression_params.bytes.get('lib', BytesCompressor.default_lib)]
        if compression_lib != blosc:
            return compression_lib.compress(blob)
        return compression_lib.compress(blob, cname=compression_params.bytes.get('cname', BytesCompressor.default_cname), \
            typesize=compression_params.bytes.get('compression_lvl', BytesCompressor.default_compression_lvl))

    @staticmethod
    def decompress(blob: bytes) -> bytes:
        compression_lib = BytesCompressor.libmap[compression_params.bytes.get('lib', BytesCompressor.default_lib)]
        if compression_lib != blosc:
            return compression_lib.decompress(blob)
        return compression_lib.decompress(blob, cname=compression_params.bytes.get('cname', BytesCompressor.default_cname), \
            typesize=compression_params.bytes.get('compression_lvl', BytesCompressor.default_compression_lvl))
