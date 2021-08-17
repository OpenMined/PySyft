from .sparse_compressor import SparseCompressor
from .dgc_compressor import DgcCompressor

registered_compressors = {
    SparseCompressor: 1,
    DgcCompressor: 2,
}

named_compressors = {
    'SparseCompressor': SparseCompressor,
    'DgcCompressor': DgcCompressor,
}