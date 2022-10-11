# future
from __future__ import annotations

# stdlib
import secrets
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

# third party
import numpy as np
import pandas as pd
import torch as th

# syft absolute
import syft as sy

# relative
from ... import lib
from ...ast.klass import pointerize_args_and_kwargs
from ...core.adp.data_subject_ledger import DataSubjectLedger
from ...util import inherit_tags
from ..common.serde.capnp import CapnpModule
from ..common.serde.capnp import chunk_bytes
from ..common.serde.capnp import combine_bytes
from ..common.serde.capnp import get_capnp_schema
from ..common.serde.capnp import serde_magic_header
from ..common.serde.serializable import serializable
from ..common.uid import UID
from ..node.abstract.node import AbstractNodeClient
from ..node.common.action import smpc_action_functions
from ..node.common.action.run_class_method_smpc_action import RunClassMethodSMPCAction

# from ..node.domain.client import DomainClient
from ..pointer.pointer import Pointer
from .ancestors import PhiTensorAncestor
from .autodp.gamma_tensor import GammaTensor
from .autodp.gamma_tensor import TensorWrappedGammaTensorPointer
from .autodp.phi_tensor import PhiTensor
from .autodp.phi_tensor import TensorWrappedPhiTensorPointer
from .config import DEFAULT_FLOAT_NUMPY_TYPE
from .config import DEFAULT_INT_NUMPY_TYPE
from .fixed_precision_tensor_ancestor import FixedPrecisionTensorAncestor
from .passthrough import PassthroughTensor  # type: ignore
from .smpc import context
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

    def _apply_tensor_op(
        self,
        other: Any,
        op_str: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        # we want to get the return type which matches the attr_path_and_name
        # so we ask lib_ast for the return type name that matches out
        # attr_path_and_name and then use that to get the actual pointer klass
        # then set the result to that pointer klass

        op = f"__{op_str}__" if op_str != "concatenate" else "concatenate"
        # remove this to dunder method before merge.
        attr_path_and_name = f"syft.core.tensor.tensor.Tensor.{op}"
        seed_id_locations = kwargs.pop("seed_id_locations", None)
        if seed_id_locations is None:
            seed_id_locations = secrets.randbits(64)

        id_at_location = smpc_action_functions.get_id_at_location_from_op(
            seed_id_locations, op
        )

        # QUESTION can the id_at_location be None?
        result_id_at_location = id_at_location

        result = TensorPointer(client=self.client)
        result.id_at_location = result_id_at_location

        if result_id_at_location is not None:
            # first downcast anything primitive which is not already PyPrimitive
            (
                downcast_args,
                downcast_kwargs,
            ) = lib.python.util.downcast_args_and_kwargs(
                args=[self, other], kwargs=kwargs
            )

            # then we convert anything which isnt a pointer into a pointer
            pointer_args, pointer_kwargs = pointerize_args_and_kwargs(
                args=downcast_args,
                kwargs=downcast_kwargs,
                client=self.client,
                gc_enabled=False,
            )

            cmd = RunClassMethodSMPCAction(
                path=attr_path_and_name,
                _self=self,
                args=pointer_args,
                kwargs=pointer_kwargs,
                id_at_location=result_id_at_location,
                seed_id_locations=seed_id_locations,
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
        **kwargs: Any,
    ) -> Union[MPCTensor, TensorPointer]:
        """Performs the operation based on op_str

        Args:
            other (Union[TensorPointer,MPCTensor,int,float,np.ndarray]): second operand.

        Returns:
            Tuple[MPCTensor,Union[MPCTensor,int,float,np.ndarray]] : Result of the operation
        """
        if isinstance(other, TensorPointer) and self.client != other.client:
            parties = [self.client, other.client]
            self_mpc = MPCTensor(secret=self, shape=self.public_shape, parties=parties)
            other_mpc = MPCTensor(
                secret=other, shape=other.public_shape, parties=parties
            )
            func = getattr(self_mpc, op_str)
            return func(other_mpc)
        elif isinstance(other, MPCTensor):
            # "self" should be secretly shared
            other_mpc, self_mpc = MPCTensor.sanity_checks(other, self)
            func = getattr(self_mpc, op_str)
            return func(other_mpc)

        return self._apply_tensor_op(other=other, op_str=op_str, **kwargs)

    def __add__(
        self,
        other: Union[TensorPointer, MPCTensor, int, float, np.ndarray],
        **kwargs: Any,
    ) -> Union[TensorPointer, MPCTensor]:
        """Apply the "add" operation between "self" and "other"

        Args:
            y (Union[TensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorPointer._apply_op(self, other, "add", **kwargs)

    def __sub__(
        self,
        other: Union[TensorPointer, MPCTensor, int, float, np.ndarray],
        **kwargs: Any,
    ) -> Union[TensorPointer, MPCTensor]:
        """Apply the "sub" operation between "self" and "other"

        Args:
            y (Union[TensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorPointer._apply_op(self, other, "sub", **kwargs)

    def __mul__(
        self,
        other: Union[TensorPointer, MPCTensor, int, float, np.ndarray],
        **kwargs: Any,
    ) -> Union[TensorPointer, MPCTensor]:
        """Apply the "mul" operation between "self" and "other"

        Args:
            y (Union[TensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorPointer._apply_op(self, other, "mul", **kwargs)

    def __matmul__(
        self,
        other: Union[TensorPointer, MPCTensor, int, float, np.ndarray],
        **kwargs: Any,
    ) -> Union[TensorPointer, MPCTensor]:
        """Apply the "matmul" operation between "self" and "other"

        Args:
            y (Union[TensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorPointer._apply_op(self, other, "matmul", **kwargs)

    def __truediv__(
        self,
        other: Union[TensorPointer, MPCTensor, int, float, np.ndarray],
        **kwargs: Any,
    ) -> Union[TensorPointer, MPCTensor]:
        """Apply the "mul" operation between "self" and "other"

        Args:
            y (Union[TensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorPointer._apply_op(self, other, "truediv", **kwargs)

    def __rtruediv__(
        self,
        other: Union[TensorPointer, MPCTensor, int, float, np.ndarray],
        **kwargs: Any,
    ) -> Union[TensorPointer, MPCTensor]:
        """Apply the "mul" operation between "self" and "other"

        Args:
            y (Union[TensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorPointer,MPCTensor] : Result of the operation.
        """
        raise NotImplementedError

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

    def concatenate(
        self,
        other: TensorPointer,
        *args: Any,
        **kwargs: Any,
    ) -> Union[TensorPointer, MPCTensor]:
        """Apply the "add" operation between "self" and "other"

        Args:
            y (Union[TensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorPointer._apply_op(self, other, "concatenate")


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
        public_dtype: Optional[str] = None,
    ) -> None:
        """data must be a list of numpy array"""

        if isinstance(child, list) or np.isscalar(child):
            child = np.array(child)

        if isinstance(child, th.Tensor):
            print(
                "Converting PyTorch tensor to numpy tensor for internal representation..."
            )
            child = child.numpy()

        # Added for convenience- might need to double check if dtype changes?
        if isinstance(child, pd.Series):
            child = child.to_numpy()

        if (
            not isinstance(child, PassthroughTensor)
            and not isinstance(child, np.ndarray)
            and not isinstance(child, GammaTensor)
        ):

            raise Exception(
                f"Data: {child} ,type: {type(child)} must be list or nd.array "
            )

        # Temp fix for windows
        if getattr(child, "dtype", None):
            if "int" in str(child.dtype) and "64" not in str(child.dtype):
                child = child.astype(DEFAULT_INT_NUMPY_TYPE)  # type: ignore
            if "float" in str(child.dtype) and "64" not in str(child.dtype):
                child = child.astype(DEFAULT_FLOAT_NUMPY_TYPE)  # type: ignore

        if not isinstance(child, (np.ndarray, PassthroughTensor, GammaTensor)) or (
            getattr(child, "dtype", None)
            not in [DEFAULT_INT_NUMPY_TYPE, DEFAULT_FLOAT_NUMPY_TYPE, np.bool_]
            and getattr(child, "dtype", None) is not None
        ):
            raise TypeError(
                "You tried to pass an a tensor of type:"
                + str(type(child))
                + " with child.dtype == "
                + str(getattr(child, "dtype", None))
                + ". Syft tensor objects only supports numpy objects of "
                + f"{DEFAULT_INT_NUMPY_TYPE,DEFAULT_FLOAT_NUMPY_TYPE,np.bool_}. "
                + "Please pass in either the supported types or change the default types in syft/core/tensor/config.py "
            )

        kwargs = {"child": child}
        super().__init__(**kwargs)

        # set public shape to be the shape of the data since we have access to it at present
        if public_shape is None:
            public_shape = tuple(self.shape)

        # set public dtype to be the dtype of the data since we have access to it at present
        if public_dtype is None:
            public_dtype = str(self.dtype)

        self.tag_name: str = ""
        self.public_shape = public_shape
        self.public_dtype = public_dtype

    def tag(self, name: str) -> Tensor:
        self.tag_name = name
        return self

    def exp(self) -> Tensor:
        if hasattr(self.child, "exp"):
            return self.__class__(self.child.exp())
        else:
            raise ValueError("Tensor Chain does not have exp function")

    def reciprocal(self) -> Tensor:
        if hasattr(self.child, "reciprocal"):
            return self.__class__(self.child.reciprocal())
        else:
            raise ValueError("Tensor Chain does not have reciprocal function")

    def softmax(self) -> Tensor:
        if hasattr(self.child, "softmax"):
            return self.__class__(self.child.softmax())
        else:
            raise ValueError("Tensor Chain does not have softmax function")

    def one_hot(self) -> Tensor:
        if hasattr(self.child, "one_hot"):
            return self.__class__(self.child.one_hot())
        else:
            raise ValueError("Tensor Chain does not have one_hot function")

    @property
    def shape(self) -> Tuple[Any, ...]:
        try:
            return self.child.shape
        except Exception:  # nosec
            return self.public_shape

    @property
    def proxy_public_kwargs(self) -> Dict[str, Any]:
        return {"public_shape": self.public_shape, "public_dtype": self.public_dtype}

    def init_pointer(
        self,
        client: Any,
        id_at_location: Optional[UID] = None,
        object_type: str = "",
        tags: Optional[List[str]] = None,
        description: str = "",
    ) -> Pointer:
        # relative

        if isinstance(self.child, PhiTensor):
            return TensorWrappedPhiTensorPointer(
                data_subjects=self.child.data_subjects,
                client=client,
                id_at_location=id_at_location,
                object_type=object_type,
                tags=tags,
                description=description,
                min_vals=self.child.min_vals,
                max_vals=self.child.max_vals,
                public_shape=getattr(self, "public_shape", None),
                public_dtype=getattr(self, "public_dtype", None),
            )
        elif isinstance(self.child, GammaTensor):
            return TensorWrappedGammaTensorPointer(
                data_subjects=self.child.data_subjects,
                client=client,
                id_at_location=id_at_location,
                object_type=object_type,
                tags=tags,
                description=description,
                min_vals=self.child.min_vals,
                max_vals=self.child.max_vals,
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

    def publish(
        self,
        get_budget_for_user: Callable,
        deduct_epsilon_for_user: Callable,
        ledger: DataSubjectLedger,
        sigma: float,
    ) -> Any:
        return self.child.publish(
            get_budget_for_user, deduct_epsilon_for_user, ledger, sigma
        )

    # TODO: remove after moving private compare to sharetensor level
    def bit_decomposition(self, ring_size: Union[int, str], bitwise: bool) -> None:
        context.tensor_values = self
        if isinstance(self.child, PhiTensor):
            self.child.child.child.bit_decomposition(ring_size, bitwise)
        else:
            self.child.bit_decomposition(ring_size, bitwise)

        return None

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

        tensor_msg.publicShape = sy.serialize(self.public_shape, to_bytes=True)

        # upcast the String class before setting to capnp
        public_dtype_func = getattr(
            self.public_dtype, "upcast", lambda: self.public_dtype
        )
        tensor_msg.publicDtype = public_dtype_func()
        tensor_msg.tagName = self.tag_name

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
            public_shape=sy.deserialize(tensor_msg.publicShape, from_bytes=True),
            public_dtype=tensor_msg.publicDtype,
        )
        tensor.tag_name = tensor_msg.tagName

        return tensor
