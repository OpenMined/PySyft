# future
from __future__ import annotations

# stdlib
from typing import Type
from typing import Union

# third party
import numpy as np
from numpy.typing import ArrayLike
import torch as th

# syft relative
from ...core.common.serde.recursive import RecursiveSerde
from ..common.serde.serializable import bind_protobuf
from .ancestors import AutogradTensorAncestor
from .ancestors import PhiTensorAncestor
from .passthrough import PassthroughTensor  # type: ignore


from typing import Optional

@bind_protobuf
class Tensor(
    PassthroughTensor, AutogradTensorAncestor, PhiTensorAncestor, RecursiveSerde
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