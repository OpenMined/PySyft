# future
from __future__ import annotations

# stdlib
import operator
from typing import Any
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

# third party
import numpy as np
import torch as th

# syft absolute
import syft as sy

# relative
from ... import lib
from ...ast.klass import pointerize_args_and_kwargs
from ...util import inherit_tags
from ..common.serde.capnp import CapnpModule
from ..common.serde.capnp import chunk_bytes
from ..common.serde.capnp import combine_bytes
from ..common.serde.capnp import get_capnp_schema
from ..common.serde.capnp import serde_magic_header
from ..common.serde.serializable import serializable
from ..common.uid import UID
from ..node.abstract.node import AbstractNodeClient
from ..node.common.action.run_class_method_action import RunClassMethodAction
from ..pointer.pointer import Pointer
from .ancestors import AutogradTensorAncestor
from .ancestors import PhiTensorAncestor
from .fixed_precision_tensor_ancestor import FixedPrecisionTensorAncestor
from .passthrough import PassthroughTensor  # type: ignore
from .smpc import utils
from .smpc.mpc_tensor import MPCTensor


class TensorPointer(Pointer):

    # Must set these at class init time despite
    # the fact that klass.Class tries to override them (unsuccessfully)
    __name__ = "TensorPointer"
    __module__ = "syft.core.tensor.tensor"

    def __init__(
        self,
        client: Any,
        id_at_location: Optional[UID] = None,
        object_type: str = "",
        tags: Optional[List[str]] = None,
        description: str = "",
        public_shape: Optional[Tuple[int, ...]] = None,
        public_dtype: Optional[Union[str, np.dtype]] = None,
    ):

        super().__init__(
            client=client,
            id_at_location=id_at_location,
            object_type=object_type,
            tags=tags,
            description=description,
        )

        self.public_shape = public_shape

        if isinstance(public_dtype, str):
            self.public_dtype = np.dtype(public_dtype)
        else:
            self.public_dtype = public_dtype

    def share(self, *parties: Tuple[AbstractNodeClient, ...]) -> MPCTensor:
        all_parties = list(parties) + [self.client]
        ring_size = utils.TYPE_TO_RING_SIZE.get(self.public_dtype, None)
        self_mpc = MPCTensor(
            secret=self,
            shape=self.public_shape,
            parties=all_parties,
            ring_size=ring_size,
        )
        return self_mpc

    def _apply_tensor_op(self, other: Any, op_str: str) -> Any:
        # we want to get the return type which matches the attr_path_and_name
        # so we ask lib_ast for the return type name that matches out
        # attr_path_and_name and then use that to get the actual pointer klass
        # then set the result to that pointer klass

        attr_path_and_name = f"syft.core.tensor.tensor.Tensor.__{op_str}__"

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
        result_public_dtype = None

        if isinstance(other, TensorPointer):
            other_shape = other.public_shape
            other_dtype = other.public_dtype
        elif isinstance(other, (int, float)):
            other_shape = (1,)
            other_dtype = np.int32
        elif isinstance(other, bool):
            other_shape = (1,)
            other_dtype = np.dtype("bool")
        elif isinstance(other, np.ndarray):
            other_shape = other.shape
            other_dtype = other.dtype

        else:
            raise ValueError(f"Invalid Type for TensorPointer:{type(other)}")

        if self.public_shape is not None and other_shape is not None:
            result_public_shape = utils.get_shape(
                op_str, self.public_shape, other_shape
            )

        if self.public_dtype is not None and other_dtype is not None:
            if self.public_dtype != other_dtype:
                raise ValueError(
                    f"Type for self and other do not match ({self.public_dtype} vs {other_dtype})"
                )
            result_public_dtype = self.public_dtype

        result.public_shape = result_public_shape
        result.public_dtype = result_public_dtype

        return result

    @staticmethod
    def _apply_op(
        self: TensorPointer,
        other: Union[TensorPointer, MPCTensor, int, float, np.ndarray],
        op_str: str,
    ) -> Union[MPCTensor, TensorPointer]:
        """Performs the operation based on op_str

        Args:
            other (Union[TensorPointer,MPCTensor,int,float,np.ndarray]): second operand.

        Returns:
            Tuple[MPCTensor,Union[MPCTensor,int,float,np.ndarray]] : Result of the operation
        """
        op = getattr(operator, op_str)

        if isinstance(other, TensorPointer) and self.client != other.client:

            parties = [self.client, other.client]
            self_mpc = MPCTensor(secret=self, shape=self.public_shape, parties=parties)
            other_mpc = MPCTensor(
                secret=other, shape=other.public_shape, parties=parties
            )
            return op(self_mpc, other_mpc)

        elif isinstance(other, MPCTensor):

            return op(other, self)

        return self._apply_tensor_op(other=other, op_str=op_str)

    def __add__(
        self, other: Union[TensorPointer, MPCTensor, int, float, np.ndarray]
    ) -> Union[TensorPointer, MPCTensor]:
        """Apply the "add" operation between "self" and "other"

        Args:
            y (Union[TensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorPointer._apply_op(self, other, "add")

    def __sub__(
        self, other: Union[TensorPointer, MPCTensor, int, float, np.ndarray]
    ) -> Union[TensorPointer, MPCTensor]:
        """Apply the "sub" operation between "self" and "other"

        Args:
            y (Union[TensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorPointer._apply_op(self, other, "sub")

    def __mul__(
        self, other: Union[TensorPointer, MPCTensor, int, float, np.ndarray]
    ) -> Union[TensorPointer, MPCTensor]:
        """Apply the "mul" operation between "self" and "other"

        Args:
            y (Union[TensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorPointer._apply_op(self, other, "mul")

    def __lt__(
        self, other: Union[TensorPointer, MPCTensor, int, float, np.ndarray]
    ) -> Union[TensorPointer, MPCTensor]:
        """Apply the "lt" operation between "self" and "other"

        Args:
            y (Union[TensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorPointer,MPCTensor] : Result of the operation.
        """

        return TensorPointer._apply_op(self, other, "lt")

    def __gt__(
        self, other: Union[TensorPointer, MPCTensor, int, float, np.ndarray]
    ) -> Union[TensorPointer, MPCTensor]:
        """Apply the "gt" operation between "self" and "other"

        Args:
            y (Union[TensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorPointer,MPCTensor] : Result of the operation.
        """

        return TensorPointer._apply_op(self, other, "gt")

    def __ge__(
        self, other: Union[TensorPointer, MPCTensor, int, float, np.ndarray]
    ) -> Union[TensorPointer, MPCTensor]:
        """Apply the "ge" operation between "self" and "other"

        Args:
            y (Union[TensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorPointer,MPCTensor] : Result of the operation.
        """

        return TensorPointer._apply_op(self, other, "ge")

    def __le__(
        self, other: Union[TensorPointer, MPCTensor, int, float, np.ndarray]
    ) -> Union[TensorPointer, MPCTensor]:
        """Apply the "le" operation between "self" and "other"

        Args:
            y (Union[TensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorPointer,MPCTensor] : Result of the operation.
        """

        return TensorPointer._apply_op(self, other, "le")

    def __eq__(  # type: ignore
        self, other: Union[TensorPointer, MPCTensor, int, float, np.ndarray]
    ) -> Union[TensorPointer, MPCTensor]:
        """Apply the "eq" operation between "self" and "other"

        Args:
            y (Union[TensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorPointer,MPCTensor] : Result of the operation.
        """

        return TensorPointer._apply_op(self, other, "eq")

    def __ne__(  # type: ignore
        self, other: Union[TensorPointer, MPCTensor, int, float, np.ndarray]
    ) -> Union[TensorPointer, MPCTensor]:
        """Apply the "ne" operation between "self" and "other"

        Args:
            y (Union[TensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorPointer,MPCTensor] : Result of the operation.
        """

        return TensorPointer._apply_op(self, other, "ne")


def to32bit(np_array: np.ndarray, verbose: bool = True) -> np.ndarray:

    if np_array.dtype == np.int64:
        if verbose:
            print("Casting internal tensor to int32")
        out = np_array.astype(np.int32)

    elif np_array.dtype == np.float64:

        if verbose:
            print("Casting internal tensor to float32")
        out = np_array.astype(np.float32)

    else:
        out = np_array

    return out


@serializable(capnp_bytes=True)
class Tensor(
    PassthroughTensor,
    AutogradTensorAncestor,
    PhiTensorAncestor,
    FixedPrecisionTensorAncestor,
    # MPCTensorAncestor,
):

    # __attr_allowlist__ = ["child", "tag_name", "public_shape", "public_dtype"]

    PointerClassOverride = TensorPointer

    def __init__(
        self,
        child: Any,
        public_shape: Optional[Tuple[int, ...]] = None,
        public_dtype: Optional[np.dtype] = None,
    ) -> None:
        """data must be a list of numpy array"""

        if isinstance(child, (list, np.int32)):
            child = to32bit(np.array(child), verbose=False)

        if isinstance(child, th.Tensor):
            print(
                "Converting PyTorch tensor to numpy tensor for internal representation..."
            )
            child = to32bit(child.numpy())

        if not isinstance(child, PassthroughTensor) and not isinstance(
            child, np.ndarray
        ):

            raise Exception(
                f"Data: {child} ,type: {type(child)} must be list or nd.array "
            )

        if not isinstance(child, (np.ndarray, PassthroughTensor)) or (
            getattr(child, "dtype", None) not in [np.int32, np.bool_]
            and getattr(child, "dtype", None) is not None
        ):
            raise TypeError(
                "You tried to pass an a tensor of type:"
                + str(type(child))
                + " with child.dtype == "
                + str(getattr(child, "dtype", None))
                + ". Syft tensor objects only support np.int32 objects at this time. Please pass in either "
                "a list of int objects or a np.int32 array. We apologise for the inconvenience and will "
                "be adding support for more types very soon!"
            )

        kwargs = {"child": child}
        super().__init__(**kwargs)

        # set public shape to be the shape of the data since we have access to it at present
        if public_shape is None:
            public_shape = tuple(self.shape)

        # set public dtype to be the dtype of the data since we have access to it at present
        if public_dtype is None:
            public_dtype = str(self.dtype)

        self.tag_name: Optional[str] = None
        self.public_shape = public_shape
        self.public_dtype = public_dtype

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
    ) -> Pointer:
        # relative
        from .autodp.single_entity_phi import SingleEntityPhiTensor
        from .autodp.single_entity_phi import TensorWrappedSingleEntityPhiTensorPointer

        # TODO:  Should create init pointer for NDimEntityPhiTensorPointer.

        if isinstance(self.child, SingleEntityPhiTensor):
            return TensorWrappedSingleEntityPhiTensorPointer(
                entity=self.child.entity,
                client=client,
                id_at_location=id_at_location,
                object_type=object_type,
                tags=tags,
                description=description,
                min_vals=self.child.min_vals,
                max_vals=self.child.max_vals,
                scalar_manager=self.child.scalar_manager,
                public_shape=getattr(self, "public_shape", None),
                public_dtype=getattr(self, "public_dtype", None),
            )
        else:
            return TensorPointer(
                client=client,
                id_at_location=id_at_location,
                object_type=object_type,
                tags=tags,
                description=description,
                public_shape=getattr(self, "public_shape", None),
                public_dtype=getattr(self, "public_dtype", None),
            )

    def bit_decomposition(self) -> Tensor:
        raise ValueError("Should not reach this point")

    def mpc_swap(self, other: Tensor) -> Tensor:
        self.child.child = other.child.child
        return self

    def _object2bytes(self) -> bytes:
        schema = get_capnp_schema(schema_file="tensor.capnp")
        tensor_struct: CapnpModule = schema.Tensor  # type: ignore
        tensor_msg = tensor_struct.new_message()

        # this is how we dispatch correct deserialization of bytes
        tensor_msg.magicHeader = serde_magic_header(type(self))

        chunk_bytes(sy.serialize(self.child, to_bytes=True), "child", tensor_msg)
        chunk_bytes(
            sy.serialize(self.public_shape, to_bytes=True), "publicShape", tensor_msg
        )
        chunk_bytes(
            sy.serialize(self.public_dtype, to_bytes=True), "publicDtype", tensor_msg
        )
        chunk_bytes(sy.serialize(self.tag_name, to_bytes=True), "tagName", tensor_msg)

        return tensor_msg.to_bytes_packed()

    @staticmethod
    def _bytes2object(buf: bytes) -> Tensor:
        schema = get_capnp_schema(schema_file="tensor.capnp")
        tensor_struct: CapnpModule = schema.Tensor  # type: ignore
        # https://stackoverflow.com/questions/48458839/capnproto-maximum-filesize
        MAX_TRAVERSAL_LIMIT = 2**64 - 1
        tensor_msg = tensor_struct.from_bytes_packed(
            buf, traversal_limit_in_words=MAX_TRAVERSAL_LIMIT
        )

        tensor = Tensor(
            child=sy.deserialize(combine_bytes(tensor_msg.child), from_bytes=True),
            public_shape=sy.deserialize(
                combine_bytes(tensor_msg.publicShape), from_bytes=True
            ),
            public_dtype=sy.deserialize(
                combine_bytes(tensor_msg.publicDtype), from_bytes=True
            ),
        )
        tensor.tag_name = sy.deserialize(
            combine_bytes(tensor_msg.tagName), from_bytes=True
        )

        return tensor
