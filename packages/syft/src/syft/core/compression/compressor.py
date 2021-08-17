# third party 
import torch as th

# relative
from .compressed_tensor import CompressedTensor
from .compression_params import compression_params
from .util import registered_compressors
from .util import named_compressors

class Compressor:
    """
    [Experimental: high-performace duet channel] accepts tensor objects and compresses using an optimum algorithm
    """

    def compress():
        pass

    def decompress():
        pass

def compress_all_possible(tensor: th.Tensor):
    compressed = CompressedTensor(tensor)
    for compressor in registered_compressors:
        compressed.compress_more(compressor)
    return compressed

def compress_configured_tensor(tensor: th.Tensor):
    if compression_params.tensor['compress']:
        compressed = CompressedTensor(tensor)
        for compressor in compression_params.tensor['compressors']:
            compressor = named_compressors[compressor]
            compressed.compress_more(compressor)
    return compressed