from .sparse_compressor import SparseCompressor

class CompressionParams:
    def __init__(self) -> None:
        self.bytes = {
            'lib': 'lzma',
            'cname': 'zlib',
            'compression_lvl': 8,
        }
        self.tensor = {
            'compress': True,
            'compressors': ['SparseCompressor'],
        }
        self.dgc_compressor = {
            'ratio': 0.8
        }
        self.connection_tested = False
        self.connection_speed = 0.0

compression_params = CompressionParams()
