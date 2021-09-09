# stdlib
from typing import Any
from typing import List

# syft absolute
from syft.core.tensor.manager import TensorChainManager

# relative
from .mpc_tensor import MPCTensor
from .utils import ispointer


class MPCTensorAncestor(TensorChainManager):
    def share(self, *parties: List[Any]) -> MPCTensor:
        # syft absolute
        from syft.core.tensor.smpc.mpc_tensor import MPCTensor

        if ispointer(self.child):
            raise ValueError(
                "Cannot call share on a remote tensor. Use MPCTensor(remote_secret)"
            )

        return MPCTensor(secret=self.child, parties=parties)
