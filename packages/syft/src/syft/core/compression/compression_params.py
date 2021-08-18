class CompressionParams:
    def __init__(self) -> None:
        self.bytes = {
            'compress': False,
            'lib': 'lzma',
            'cname': 'zlib',
            'compression_lvl': 8,
        }
        self.tensor = {
            'compress': True,
            'compressors': ['DgcCompressor'],
        }
        self.dgc_compressor = {
            'ratio': 0.8
        }
        self.deep_reduce = {
            'compress_ratio': 0.5, 
            'deepreduce':'index', 
            'index':'bloom',
        }
        self.connection_tested = False
        self.connection_speed = 0.0

compression_params = CompressionParams()
