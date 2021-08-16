from .sparse_compressor import SparseCompressor
from .util import registered_compressors

class CompressionParams:
    def __init__(self) -> None:
        self.bytes = {
            'lib': 'lzma',
            'cname': 'zlib',
            'compression_lvl': 8,
        }
        self.tensor = {
            'compress': True,
            'compressors': SparseCompressor,
        }
        self.connection_tested = False
        self.connection_speed = 0.0
        self.grad_compressors = []

compression_params = CompressionParams()
