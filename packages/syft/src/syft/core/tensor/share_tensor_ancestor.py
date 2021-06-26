# syft absolute
from syft.core.tensor.manager import TensorChainManager
from syft.core.tensor.share_tensor import ShareTensor


def is_pointer(val):
    if "Pointer" in val.__name__:
        return True


class ShareTensorAncestor(TensorChainManager):
    def share(self, parties, shape=None):
        if is_pointer in self.child:
            # Remote secret
            if shape is None:
                raise ValueError("Shape must be specified when the secret is remote")

        # TODO
        return self

    def reconstruct(self):
        if not isinstance(self, ShareTensor):
            raise ValueError("Child should be a list of values (remote or local)")

        # TODO
        return None
