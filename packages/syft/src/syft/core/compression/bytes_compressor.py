import lzma
import blosc

from .compression_flags import compression_params

class BytesCompressor:
    """
    [Experimental: high-performace duet channel] base class for specialized compression algorithms
    """
    libmap = {
        'lzma': lzma,
        'blosc': blosc,
    }
    default_lib = 'lzma'
    default_cname = 'zlib'
    default_compression_lvl = 8

    @staticmethod
    def compress(blob: bytes) -> bytes:
        compression_lib = BytesCompressor.libmap[compression_params.bytes.get('lib', BytesCompressor.default_lib)]
        header = str(compression_params.bytes.get('lib', BytesCompressor.default_lib)) + '|' + str(compression_params.bytes.get('cname', BytesCompressor.default_cname)) \
            + '|' + str(compression_params.bytes.get('compression_lvl', BytesCompressor.default_compression_lvl)) + '|'
        if compression_lib != blosc:
            return header.encode('utf-8') + compression_lib.compress(blob)

        return header.encode('utf-8') + compression_lib.compress(blob, cname=compression_params.bytes.get('cname', BytesCompressor.default_cname), \
            typesize=compression_params.bytes.get('compression_lvl', BytesCompressor.default_compression_lvl))

    @staticmethod
    def decompress(blob: bytes) -> bytes:
        is_compressed = False
        for libkey in BytesCompressor.libmap:
            if(blob[len(libkey)] == 124):
                is_compressed = True
                break
        
        if not is_compressed:
            return blob

        blob = blob.split(b'|', 3)

        compression_lib = BytesCompressor.libmap[blob[0].decode('utf-8')]
        print(compression_lib, blob[3], str(blob[1]), int(blob[2]))
        if compression_lib != blosc:
            return compression_lib.decompress(blob[3])

        return compression_lib.decompress(blob[3], cname=blob[1].decode('utf-8'), typesize=int(blob[2].decode('utf-8')))
