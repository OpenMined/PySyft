# syft absolute
from syft.core.tensor.manager import TensorChainManager
from syft.core.tensor.share_tensor import ShareTensor


class ShareTensorAncestor(TensorChainManager):
    def generate_przs(self, shape, rank):
        print(self)
        share = self.child
        share_1 = share.generators_przs[0].integers(
            low=self.min_value, high=self.max_value
        )
        share_2 = share.generators_przs[1].integers(
            low=self.min_value, high=self.max_value
        )

        if share.child is None:
            share.child = share_1 - share_2
        else:
            share.child += share_1 - share_2
        return share
