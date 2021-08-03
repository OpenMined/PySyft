# future
from __future__ import annotations

# stdlib
from typing import Optional

# third party
import numpy as np
import torch as th

# relative
# syft relative
from ...core.common.serde.recursive import RecursiveSerde
from ..common.serde.serializable import bind_protobuf
from .ancestors import AutogradTensorAncestor
from .ancestors import PhiTensorAncestor
from .passthrough import PassthroughTensor  # type: ignore
from .fixed_precision_tensor_ancestor import FixedPrecisionTensorAncestor
from .smpc.mpc_tensor_ancestor import MPCTensorAncestor
# from .smpc.share_tensor import ShareTensor


# TODO: Need to double check to see if smpc.ShareTensor operations are working correctly here since it's not inherited
@bind_protobuf
class Tensor(
    PassthroughTensor, AutogradTensorAncestor, PhiTensorAncestor, FixedPrecisionTensorAncestor, MPCTensorAncestor,
    RecursiveSerde
):

    __attr_allowlist__ = ["child"]

    def __init__(self, child):
        """data must be a list of numpy array"""

        if isinstance(child, list):
            child = np.array(child)

        if isinstance(child, th.Tensor):
            child = child.numpy()

        if not isinstance(child, PassthroughTensor) and not isinstance(
            child, np.ndarray
        ):
            raise Exception("Data must be list or nd.array")

        super().__init__(child=child)

        self.tag_name: Optional[str] = None

    def tag(self, name: str) -> None:
        self.tag_name = name
