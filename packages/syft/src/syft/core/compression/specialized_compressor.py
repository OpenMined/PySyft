import torch as th

class SpecializedCompressor:
    """
    [Experimental: high-performace duet channel] base class for specialized compression algorithms
    """
    @staticmethod
    def is_eligible(tensor: th.Tensor):
        raise NotImplementedError()

    @staticmethod
    def compress(tensor: th.Tensor):
        raise NotImplementedError()

    @staticmethod
    def decompress(tensor: th.Tensor):
        raise NotImplementedError()
