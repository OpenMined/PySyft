# future
from __future__ import annotations

# stdlib
from typing import Any, List
from typing import Optional

# third party
import numpy as np
import torch as th

# relative
# syft relative
from syft.core.common import UID
from syft.core.pointer.pointer import Pointer

from ...core.common.serde.recursive import RecursiveSerde
from ..common.serde.serializable import bind_protobuf
from .ancestors import AutogradTensorAncestor
from .ancestors import PhiTensorAncestor
from .fixed_precision_tensor_ancestor import FixedPrecisionTensorAncestor
from .passthrough import PassthroughTensor  # type: ignore

# from .smpc.share_tensor import ShareTensor

class TensorPointer(Pointer):

    # Must set these at class init time despite
    # the fact that klass.Class tries to override them (unsuccessfully)
    __name__ = "TensorPointer"
    __module__ = "syft.proxy.syft.core.tensor.tensor"

    def __init__(self,
                 client: Any,
                 id_at_location: Optional[UID] = None,
                 object_type: str = "",
                 tags: Optional[List[str]] = None,
                 description: str = "",
                 ):
        super().__init__(client=client,
                         id_at_location=id_at_location,
                         object_type=object_type,
                         tags=tags,
                         description=description)



# TODO: Need to double check to see if smpc.ShareTensor operations are working correctly here since it's not inherited
@bind_protobuf
class Tensor(
    PassthroughTensor,
    AutogradTensorAncestor,
    PhiTensorAncestor,
    FixedPrecisionTensorAncestor,
    # MPCTensorAncestor,
    RecursiveSerde,
):

    __attr_allowlist__ = ["child", "tag_name"]

    PointerClassOverride = TensorPointer

    def __init__(self, child: Any) -> None:
        """data must be a list of numpy array"""

        if isinstance(child, list):
            child = np.array(child)

        if isinstance(child, th.Tensor):
            child = child.numpy()

        if not isinstance(child, PassthroughTensor) and not isinstance(
            child, np.ndarray
        ):
            raise Exception("Data must be list or nd.array")

        kwargs = {"child": child}
        super().__init__(**kwargs)
        self.tag_name: Optional[str] = None


    def tag(self, name: str) -> Tensor:
        self.tag_name = name
        return self


    def init_pointer(self,
                 client: Any,
                 id_at_location: Optional[UID] = None,
                 object_type: str = "",
                 tags: Optional[List[str]] = None,
                 description: str = "",
                 ):
        return TensorPointer(client=client,
                             id_at_location=id_at_location,
                             object_type=object_type,
                             tags=tags,
                             description=description)