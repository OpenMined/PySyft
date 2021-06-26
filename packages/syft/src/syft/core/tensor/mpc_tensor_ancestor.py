# syft absolute
from syft.core.tensor.manager import TensorChainManager
from syft.core.tensor.mpc_tensor import MPCTensor
from syft.core.tensor.share_tensor import ShareTensor


def is_pointer(val):
    if "Pointer" in val.__name__:
        return True


class MPCTensorAncestor(TensorChainManager):
    def share(self, parties, shape=None):
        if is_pointer in self.child:
            # Remote secret
            if shape is None:
                raise ValueError("Shape must be specified when the secret is remote")
            self._remote_share(parties, shape)
        else:
            self._local_share(parties)

        return self

    def _remote_share(self, parties, shape):
        ...

    def _local_share(self, parties):
        # TODO: ShareTensor needs to have serde serializer/deserializer
        shares = ShareTensor.generate_shares(secret=self.child, nr_shares=len(parties))
        self.child = [share.send(party) for share, party in zip(shares, parties)]
