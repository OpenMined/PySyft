# future
from __future__ import annotations

# stdlib
from typing import Any
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

# third party
import numpy as np
import torch as th

# relative
from ... import lib
from ...ast.klass import pointerize_args_and_kwargs
from ...core.common.serde.recursive import RecursiveSerde
from ...util import inherit_tags
from ..common.serde.serializable import bind_protobuf
from ..common.uid import UID
from ..node.abstract.node import AbstractNodeClient
from ..node.common.action.run_class_method_action import RunClassMethodAction
from ..pointer.pointer import Pointer
from .ancestors import AutogradTensorAncestor
from .ancestors import PhiTensorAncestor
from .fixed_precision_tensor_ancestor import FixedPrecisionTensorAncestor
from .passthrough import PassthroughTensor
from .smpc.mpc_tensor import MPCTensor


class TensorPointer(Pointer):

    # Must set these at class init time despite
    # the fact that klass.Class tries to override them (unsuccessfully)
    __name__ = "TensorPointer"
    __module__ = "syft.proxy.syft.core.tensor.tensor"

    def __init__(
        self,
        client: Any,
        id_at_location: Optional[UID] = None,
        object_type: str = "",
        tags: Optional[List[str]] = None,
        description: str = "",
        public_shape: Optional[Tuple[int, ...]] = None,
    ):

        super().__init__(
            client=client,
            id_at_location=id_at_location,
            object_type=object_type,
            tags=tags,
            description=description,
        )

        self.public_shape = public_shape

    def share(self, *parties: Tuple[AbstractNodeClient, ...]) -> MPCTensor:

        parties = list(parties) + [self.client]

        self_mpc = MPCTensor(secret=self, shape=self.public_shape, parties=parties)

        return self_mpc

    def simple_add(self, other: Any) -> TensorPointer:
        # we want to get the return type which matches the attr_path_and_name
        # so we ask lib_ast for the return type name that matches out
        # attr_path_and_name and then use that to get the actual pointer klass
        # then set the result to that pointer klass

        attr_path_and_name = "syft.core.tensor.tensor.Tensor.__add__"

        result = TensorPointer(client=self.client)

        # QUESTION can the id_at_location be None?
        result_id_at_location = getattr(result, "id_at_location", None)

        if result_id_at_location is not None:
            # first downcast anything primitive which is not already PyPrimitive
            (
                downcast_args,
                downcast_kwargs,
            ) = lib.python.util.downcast_args_and_kwargs(args=[other], kwargs={})

            # then we convert anything which isnt a pointer into a pointer
            pointer_args, pointer_kwargs = pointerize_args_and_kwargs(
                args=downcast_args,
                kwargs=downcast_kwargs,
                client=self.client,
                gc_enabled=False,
            )

            cmd = RunClassMethodAction(
                path=attr_path_and_name,
                _self=self,
                args=pointer_args,
                kwargs=pointer_kwargs,
                id_at_location=result_id_at_location,
                address=self.client.address,
            )
            self.client.send_immediate_msg_without_reply(msg=cmd)

        inherit_tags(
            attr_path_and_name=attr_path_and_name,
            result=result,
            self_obj=self,
            args=[other],
            kwargs={},
        )

        result_public_shape = None

        if self.public_shape is not None and other.public_shape is not None:
            result_public_shape = (
                np.empty(self.public_shape) + np.empty(other.public_shape)
            ).shape

        result.public_shape = result_public_shape

        return result

    def __add__(self, other: Any) -> Union[TensorPointer, MPCTensor]:

        if self.client != other.client:

            parties = [self.client, other.client]

            self_mpc = MPCTensor(secret=self, shape=self.public_shape, parties=parties)
            other_mpc = MPCTensor(
                secret=other, shape=other.public_shape, parties=parties
            )

            print(self_mpc.__repr__())
            print(other_mpc.__repr__())

            return self_mpc + other_mpc

        return self.simple_add(other=other)


# TODO: Need to double check to see if smpc.ShareTensor operations are working correctly
# here since it's not inherited
@bind_protobuf
class Tensor(
    PassthroughTensor,
    AutogradTensorAncestor,
    PhiTensorAncestor,
    FixedPrecisionTensorAncestor,
    # MPCTensorAncestor,
    RecursiveSerde,
):

    __attr_allowlist__ = ["child", "tag_name", "public_shape"]

    PointerClassOverride = TensorPointer

    def __init__(
        self, child: Any, public_shape: Optional[Tuple[int, ...]] = None
    ) -> None:
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
        self.public_shape = public_shape

    def tag(self, name: str) -> Tensor:
        self.tag_name = name
        return self

    def init_pointer(
        self,
        client: Any,
        id_at_location: Optional[UID] = None,
        object_type: str = "",
        tags: Optional[List[str]] = None,
        description: str = "",
    ):

        return TensorPointer(
            client=client,
            id_at_location=id_at_location,
            object_type=object_type,
            tags=tags,
            description=description,
            public_shape=getattr(self, "public_shape", None),
        )
