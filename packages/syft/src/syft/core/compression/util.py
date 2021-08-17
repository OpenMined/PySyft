from .sparse_compressor import SparseCompressor
from .dgc_compressor import DgcCompressor
from .compression_params import compression_params

registered_compressors = {
    SparseCompressor: 1,
    DgcCompressor: 2,
}

named_compressors = {
    'SparseCompressor': SparseCompressor,
    'DgcCompressor': DgcCompressor,
}

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