import torch as th

from .specialized_compressor import SpecializedCompressor

class SparseCompressor(SpecializedCompressor):
    """
    [Experimental: high-performace duet channel] process sparse tensors
    """
    @staticmethod
    def is_eligible(tensor: th.Tensor):
        return tensor.numel() > 10 and tensor.to_sparse().values().numel() * 2 < tensor.numel()

    def compress(self, tensor: th.Tensor) -> th.Tensor:
        raw_saprse = tensor.to_sparse()
        res1 = th.cat((raw_saprse.indices(), th.Tensor(list(raw_saprse.size())).reshape(-1, 1)), dim=1)
        res2 = th.cat((res1, th.cat((raw_saprse.values(), th.Tensor([0]))).reshape(1, -1)), dim=0)
        return res2

    def decompress(self, tensor: th.Tensor) -> th.Tensor:
        return th.sparse_coo_tensor(indices=tensor[:-1, :-1], values=tensor[-1, :][:-1], size=tensor[:, -1][:-1].int().tolist()).to_dense()