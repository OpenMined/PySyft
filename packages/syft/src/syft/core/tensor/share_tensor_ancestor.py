# syft absolute
from syft.core.tensor.manager import TensorChainManager
from syft.core.tensor.share_tensor import ShareTensor
from syft.core.tensor.fixed_precision_tensor import FixedPrecisionTensor


class ShareTensorAncestor(TensorChainManager):
    def generate_przs(self, shape, rank):
        fpt = self
        share = self.child
        if not isinstance(share, ShareTensor):
            fpt = FixedPrecisionTensor(value=share)
            fpt.child = ShareTensor(value=fpt.child, rank = rank)
            share = fpt.child

        share_1 = share.generators_przs[0].integers(
            low=share.min_value, high=share.max_value
        )
        share_2 = share.generators_przs[1].integers(
            low=share.min_value, high=share.max_value
        )

        if share.child is None:
            share.child = share_1 - share_2
        else:
            share.child += share_1 - share_2
        return fpt
