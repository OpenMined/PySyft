# third party 
import torch as th

# relative
from .compressed_tensor import CompressedTensor
from .util import registered_compressors

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