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

def pack_grace(values, indices, size):
    res1 = th.cat((values, th.Tensor([0]*(len(size)-1) + [len(size)])))
    res2 = th.cat((indices, th.Tensor(list(size))))
    return th.cat((res1.reshape(1, -1), res2.reshape(1, -1)), dim=0)

def unpack_grace(packed):
    size_len = int(packed[0, -1])
    size = th.Size(packed[1, -size_len:].int().tolist())
    values = packed[0, :-size_len]
    indices = packed[1, :-size_len]

    return (values, indices), size

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