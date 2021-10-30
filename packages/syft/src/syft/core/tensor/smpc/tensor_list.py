"""CryptoStore manages the needed crypto primitives."""
# future
from __future__ import annotations

# stdlib
from typing import Any
from typing import List
from typing import TYPE_CHECKING
from typing import Union

# relative
from ...common.serde.serializable import serializable
from .share_tensor import ShareTensor

if TYPE_CHECKING:
    # relative
    from .... import Tensor


@serializable(recursive_serde=True)
class TensorList:
    """Custom Tensor List Class for indexing operations.
    Temporary class to replace python list as
    __len__ and resolve pointer type are not working.
    """

    __slots__ = ("list",)
    __attr_allowlist__ = ("list",)

    def __init__(self) -> None:
        """Initializer."""
        self.list: List[Any] = []

    def append(self, data: Union[TensorList, ShareTensor, Tensor]) -> None:
        # relative
        from .... import Tensor

        if not isinstance(data, (TensorList, ShareTensor, Tensor)):
            raise ValueError("Input data should be ShareTensor or TensorList or Tensor")
        self.list.append(data)

    def __getitem__(self, item: int) -> ShareTensor:
        return self.list[item]

    def get_tensor_pointer(self, item: int) -> Tensor:
        return self.list[item]

    def get_tensor_list(self, item: int) -> TensorList:
        return self.list[item]

    def __repr__(self) -> str:
        str_list = ""
        for data in self.list:
            str_list += data.__repr__() + ","
        return f"[{str_list}]"
