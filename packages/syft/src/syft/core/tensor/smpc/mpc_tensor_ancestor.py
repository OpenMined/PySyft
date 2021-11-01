# stdlib
from typing import Any
from typing import List

# relative
from ..manager import TensorChainManager
from .mpc_tensor import MPCTensor
from .utils import ispointer


class MPCTensorAncestor(TensorChainManager):
    def share(self, *parties: List[Any]) -> MPCTensor:
        # relative
        from .mpc_tensor import MPCTensor

        if ispointer(self.child):
            raise ValueError(
                "Cannot call share on a remote tensor. Use MPCTensor(remote_secret)"
            )

        return MPCTensor(secret=self.child, parties=list(parties))
