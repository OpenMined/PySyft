# syft absolute
from syft.core.tensor.manager import TensorChainManager
from syft.core.tensor.smpc.share_tensor import ShareTensor


def is_pointer(val):
    if "Pointer" in type(val).__name__:
        return True


class MPCTensorAncestor(TensorChainManager):
    def share(self, *parties):
        # syft absolute
        from syft.core.tensor.mpc_tensor import MPCTensor

        if is_pointer(self.child):
            raise ValueError(
                "Cannot call share on a remote tensor. Use MPCTensor(remote_secret)"
            )

        return MPCTensor(secret=self.child, parties=parties)
