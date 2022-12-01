# future
from __future__ import annotations

# stdlib
from collections.abc import Sequence
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

# third party
import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray
from scipy.ndimage.interpolation import rotate

# relative
from .... import lib
from ....ast.klass import pointerize_args_and_kwargs
from ....core.adp.data_subject import DataSubject
from ....core.adp.data_subject_ledger import DataSubjectLedger
from ....core.adp.data_subject_list import DataSubjectArray
from ....core.adp.data_subject_list import dslarraytonumpyutf8
from ....core.adp.data_subject_list import numpyutf8todslarray
from ....core.node.common.action.get_or_set_property_action import (
    GetOrSetPropertyAction,
)
from ....core.node.common.action.get_or_set_property_action import PropertyActions
from ....lib.numpy.array import capnp_deserialize
from ....lib.numpy.array import capnp_serialize
from ....lib.python.util import upcast
from ....util import inherit_tags
from ...common.serde.capnp import CapnpModule
from ...common.serde.capnp import chunk_bytes
from ...common.serde.capnp import combine_bytes
from ...common.serde.capnp import get_capnp_schema
from ...common.serde.capnp import serde_magic_header
from ...common.serde.deserialize import _deserialize as deserialize
from ...common.serde.serializable import serializable
from ...common.serde.serialize import _serialize as serialize
from ...common.uid import UID
from ...node.abstract.node import AbstractNodeClient
from ...node.common.action.run_class_method_action import RunClassMethodAction
from ...node.enums import PointerStatus
from ...pointer.pointer import Pointer
from ..broadcastable import is_broadcastable
from ..config import DEFAULT_INT_NUMPY_TYPE
from ..fixed_precision_tensor import FixedPrecisionTensor
from ..lazy_repeat_array import compute_min_max
from ..lazy_repeat_array import lazyrepeatarray
from ..passthrough import AcceptableSimpleType  # type: ignore
from ..passthrough import PassthroughTensor  # type: ignore
from ..passthrough import SupportedChainType  # type: ignore
from ..passthrough import is_acceptable_simple_type  # type: ignore
from ..smpc import utils
from ..smpc.mpc_tensor import MPCTensor
from ..smpc.utils import TYPE_TO_RING_SIZE
from ..util import implements
from .adp_tensor import ADPTensor
from .gamma_tensor import GammaTensor
from .gamma_tensor import TensorWrappedGammaTensorPointer

INPLACE_OPS = {"resize", "sort"}


@serializable(recursive_serde=True)
class TensorWrappedPhiTensorPointer(Pointer):
    __name__ = "TensorWrappedPhiTensorPointer"
    __module__ = "syft.core.tensor.autodp.phi_tensor"
    __attr_allowlist__ = [
        # default pointer attrs
        "client",
        "id_at_location",
        "object_type",
        "tags",
        "description",
        # phi_tensor attrs
        "data_subjects",
        "min_vals",
        "max_vals",
        "public_dtype",
        "public_shape",
    ]

    __serde_overrides__ = {
        "client": [lambda x: x.address, lambda y: y],
        "public_shape": [lambda x: x, lambda y: upcast(y)],
        "data_subjects": [dslarraytonumpyutf8, numpyutf8todslarray],
        "public_dtype": [lambda x: str(x), lambda y: np.dtype(y)],
    }
    _exhausted = False
    is_enum = False
    PUBLISH_POINTER_TYPE = "numpy.ndarray"

    def __init__(
        self,
        data_subjects: np.ndarray,
        min_vals: np.typing.ArrayLike,
        max_vals: np.typing.ArrayLike,
        client: Any,
        id_at_location: Optional[UID] = None,
        object_type: str = "",
        tags: Optional[List[str]] = None,
        description: str = "",
        public_shape: Optional[Tuple[int, ...]] = None,
        public_dtype: Optional[np.dtype] = None,
    ):
        super().__init__(
            client=client,
            id_at_location=id_at_location,
            object_type=object_type,
            tags=tags,
            description=description,
        )

        self.min_vals = min_vals
        self.max_vals = max_vals
        self.data_subjects = data_subjects
        self.public_shape = public_shape
        self.public_dtype = public_dtype

    # TODO: Modify for large arrays
    @property
    def synthetic(self) -> np.ndarray:
        public_dtype_func = getattr(
            self.public_dtype, "upcast", lambda: self.public_dtype
        )

        return (
            np.random.rand(*list(self.public_shape))  # type: ignore
            * (self.max_vals.data - self.min_vals.data)
            + self.min_vals.data
        ).astype(public_dtype_func())

    def __repr__(self) -> str:
        repr_string = f"PointerId: {self.id_at_location.no_dash}"
        if hasattr(self.client, "obj_exists"):
            _ptr_status = (
                PointerStatus.READY.value
                if self.exists
                else PointerStatus.PROCESSING.value
            )
            repr_string += f"\nStatus: {_ptr_status}"
        repr_string += f"\nRepresentation: {self.synthetic.__repr__()}"
        repr_string += "\n\n(The data printed above is synthetic - it's an imitation of the real data.)"
        return repr_string

    def share(self, *parties: Tuple[AbstractNodeClient, ...]) -> MPCTensor:
        all_parties = list(parties) + [self.client]
        ring_size = TYPE_TO_RING_SIZE.get(self.public_dtype, None)
        self_mpc = MPCTensor(
            secret=self,
            shape=self.public_shape,
            ring_size=ring_size,
            parties=all_parties,
        )
        return self_mpc

    @property
    def shape(self) -> Optional[Tuple[int, ...]]:
        if hasattr(self, "public_shape"):
            return self.public_shape
        else:
            return None

    def _apply_tensor_op(self, other: Any, op_str: str) -> Any:
        # we want to get the return type which matches the attr_path_and_name
        # so we ask lib_ast for the return type name that matches out
        # attr_path_and_name and then use that to get the actual pointer klass
        # then set the result to that pointer klass

        # We always maintain a Tensor hierarchy Tensor ---> PT--> Actual Data
        attr_path_and_name = f"syft.core.tensor.tensor.Tensor.{op_str}"

        min_vals, max_vals = compute_min_max(
            self.min_vals, self.max_vals, other, op_str
        )

        result = TensorWrappedPhiTensorPointer(
            data_subjects=self.data_subjects,
            min_vals=min_vals,
            max_vals=max_vals,
            client=self.client,
        )

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

        if isinstance(other, TensorWrappedPhiTensorPointer):
            other_shape = other.public_shape
            other_dtype = other.public_dtype
        elif isinstance(other, (int, float)):
            other_shape = (1,)
            other_dtype = DEFAULT_INT_NUMPY_TYPE
        elif isinstance(other, bool):
            other_shape = (1,)
            other_dtype = np.dtype("bool")
        elif isinstance(other, np.ndarray):
            other_shape = other.shape
            other_dtype = other.dtype
        else:
            raise ValueError(
                f"Invalid Type for TensorWrappedPhiTensorPointer:{type(other)}"
            )

        if self.public_shape is not None and other_shape is not None:
            result_public_shape = utils.get_shape(
                op_str, self.public_shape, other_shape
            )
        if self.public_dtype is None or other_dtype is None:
            if self.public_dtype != other_dtype:
                raise ValueError(
                    f"Dtype for self: {self.public_dtype} and other :{other_dtype} should not be None"
                )
        result_public_dtype = self.public_dtype

        result.public_shape = result_public_shape
        result.public_dtype = result_public_dtype

        result.client.processing_pointers[result.id_at_location] = True

        return result

    def _apply_self_tensor_op(self, op_str: str, *args: Any, **kwargs: Any) -> Any:
        # we want to get the return type which matches the attr_path_and_name
        # so we ask lib_ast for the return type name that matches out
        # attr_path_and_name and then use that to get the actual pointer klass
        # then set the result to that pointer klass

        # We always maintain a Tensor hierarchy Tensor ---> PT--> Actual Data
        attr_path_and_name = f"syft.core.tensor.tensor.Tensor.{op_str}"

        min_vals, max_vals = compute_min_max(
            self.min_vals, self.max_vals, None, op_str, *args, **kwargs
        )

        if hasattr(self.data_subjects, op_str):
            if op_str == "choose":
                if kwargs == {}:
                    mode = None
                    for arg in args[1:]:
                        if isinstance(arg, str):
                            mode = arg
                            break
                    if mode is None:
                        if isinstance(
                            args[0],
                            (
                                TensorWrappedPhiTensorPointer,
                                TensorWrappedGammaTensorPointer,
                            ),
                        ):
                            data_subjects = np.array(
                                np.choose(
                                    np.ones(args[0].shape, dtype=np.int64),
                                    self.data_subjects,
                                )
                            )
                        else:
                            data_subjects = np.array(
                                np.choose(args[0], self.data_subjects)
                            )
                    else:
                        if isinstance(
                            args[0],
                            (
                                TensorWrappedPhiTensorPointer,
                                TensorWrappedGammaTensorPointer,
                            ),
                        ):
                            data_subjects = np.array(
                                np.choose(
                                    np.ones(args[0].shape, dtype=np.int64),
                                    self.data_subjects,
                                    mode=mode,
                                )
                            )
                        else:
                            data_subjects = np.array(
                                np.choose(args[0], self.data_subjects, mode=mode)
                            )
                else:
                    data_subjects = np.choose(
                        kwargs["choices"], self.data_subjects, kwargs["mode"]
                    )
            else:
                data_subjects = getattr(self.data_subjects, op_str)(*args, **kwargs)
            if op_str in INPLACE_OPS:
                data_subjects = self.data_subjects
        elif op_str in ("ones_like", "zeros_like"):
            data_subjects = self.data_subjects
        else:
            raise ValueError(f"Invalid Numpy Operation: {op_str} for DSA")

        result = TensorWrappedPhiTensorPointer(
            data_subjects=data_subjects,
            min_vals=min_vals,
            max_vals=max_vals,
            client=self.client,
        )

        # QUESTION can the id_at_location be None?
        result_id_at_location = getattr(result, "id_at_location", None)

        if result_id_at_location is not None:
            # first downcast anything primitive which is not already PyPrimitive
            (
                downcast_args,
                downcast_kwargs,
            ) = lib.python.util.downcast_args_and_kwargs(args=args, kwargs=kwargs)

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
            args=args,
            kwargs=kwargs,
        )
        if op_str == "choose":
            dummy_res = np.ones(self.public_shape, dtype=np.int64)
            if isinstance(
                args[0],
                (TensorWrappedPhiTensorPointer, TensorWrappedGammaTensorPointer),
            ):
                temp_args = (np.ones(args[0].shape, dtype=np.int64), *args[1:])
                dummy_res = getattr(dummy_res, op_str)(*temp_args, **kwargs)
            else:
                dummy_res = getattr(dummy_res, op_str)(*args, **kwargs)
        else:
            dummy_res = np.empty(self.public_shape)
            if hasattr(dummy_res, op_str):
                if op_str in INPLACE_OPS:
                    getattr(dummy_res, op_str)(*args, **kwargs)
                else:
                    dummy_res = getattr(dummy_res, op_str)(*args, **kwargs)
            elif hasattr(np, op_str):
                dummy_res = getattr(np, op_str)(dummy_res, *args, *kwargs)
            else:
                raise ValueError(f"Invalid Numpy Operation: {op_str} for Pointer")

        result.public_shape = dummy_res.shape
        result.public_dtype = dummy_res.dtype

        result.client.processing_pointers[result.id_at_location] = True

        return result

    @property
    def gamma(self) -> TensorWrappedGammaTensorPointer:
        return TensorWrappedGammaTensorPointer(
            data_subjects=self.data_subjects,
            client=self.client,
            id_at_location=self.id_at_location,
            object_type=self.object_type,
            tags=self.tags,
            description=self.description,
            min_vals=self.min_vals,
            max_vals=self.max_vals,
            public_shape=getattr(self, "public_shape", None),
            public_dtype=getattr(self, "public_dtype", None),
        )

    def copy(self, *args: Any, **kwargs: Any) -> TensorWrappedPhiTensorPointer:
        return self._apply_self_tensor_op("copy", *args, **kwargs)

    def round(self, *args: Any, **kwargs: Any) -> TensorWrappedPhiTensorPointer:
        return self._apply_self_tensor_op("round", *args, **kwargs)

    def __round__(self, *args: Any, **kwargs: Any) -> TensorWrappedPhiTensorPointer:
        return self.round(*args, **kwargs)

    @staticmethod
    def _apply_op(
        self: TensorWrappedPhiTensorPointer,
        other: Union[TensorWrappedPhiTensorPointer, MPCTensor, int, float, np.ndarray],
        op_str: str,
    ) -> Union[MPCTensor, TensorWrappedPhiTensorPointer]:
        """Performs the operation based on op_str

        Args:
            other (Union[TensorWrappedPhiTensorPointer,MPCTensor,int,float,np.ndarray]): second operand.

        Returns:
            Tuple[MPCTensor,Union[MPCTensor,int,float,np.ndarray]] : Result of the operation
        """
        if isinstance(other, TensorWrappedPhiTensorPointer):
            if np.array(self.data_subjects != other.data_subjects).all():  # type: ignore
                return getattr(self.gamma, op_str)(other.gamma)
        elif isinstance(other, TensorWrappedGammaTensorPointer):
            return getattr(self.gamma, op_str)(other)

        if (
            isinstance(other, TensorWrappedPhiTensorPointer)
            and self.client != other.client
        ):

            parties = [self.client, other.client]

            self_mpc = MPCTensor(secret=self, shape=self.public_shape, parties=parties)
            other_mpc = MPCTensor(
                secret=other, shape=other.public_shape, parties=parties
            )

            return getattr(self_mpc, op_str)(other_mpc)

        elif isinstance(other, MPCTensor):

            return getattr(other, op_str)(self)
        elif is_acceptable_simple_type(other) or isinstance(
            other, TensorWrappedPhiTensorPointer
        ):
            return self._apply_tensor_op(other=other, op_str=op_str)
        else:
            print("Type is unsupported:" + str(type(other)))
            raise NotImplementedError

    def __add__(
        self,
        other: Union[TensorWrappedPhiTensorPointer, MPCTensor, int, float, np.ndarray],
    ) -> Union[TensorWrappedPhiTensorPointer, MPCTensor]:
        """Apply the "add" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedPhiTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedPhiTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedPhiTensorPointer._apply_op(self, other, "__add__")

    def __sub__(
        self,
        other: Union[TensorWrappedPhiTensorPointer, MPCTensor, int, float, np.ndarray],
    ) -> Union[TensorWrappedPhiTensorPointer, MPCTensor]:
        """Apply the "sub" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedPhiTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedPhiTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedPhiTensorPointer._apply_op(self, other, "__sub__")

    def __mul__(
        self,
        other: Union[TensorWrappedPhiTensorPointer, MPCTensor, int, float, np.ndarray],
    ) -> Union[TensorWrappedPhiTensorPointer, MPCTensor]:
        """Apply the "mul" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedPhiTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedPhiTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedPhiTensorPointer._apply_op(self, other, "__mul__")

    def __matmul__(
        self,
        other: Union[TensorWrappedPhiTensorPointer, MPCTensor, int, float, np.ndarray],
    ) -> Union[TensorWrappedPhiTensorPointer, MPCTensor]:
        """Apply the "matmul" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedPhiTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedPhiTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedPhiTensorPointer._apply_op(self, other, "__matmul__")

    def __rmatmul__(
        self,
        other: Union[TensorWrappedPhiTensorPointer, MPCTensor, int, float, np.ndarray],
    ) -> Union[TensorWrappedPhiTensorPointer, MPCTensor]:
        """Apply the "rmatmul" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedPhiTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedPhiTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedPhiTensorPointer._apply_op(self, other, "__rmatmul__")

    def __lt__(
        self,
        other: Union[TensorWrappedPhiTensorPointer, MPCTensor, int, float, np.ndarray],
    ) -> Union[TensorWrappedPhiTensorPointer, MPCTensor]:
        """Apply the "lt" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedPhiTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedPhiTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedPhiTensorPointer._apply_op(self, other, "__lt__")

    def __gt__(
        self,
        other: Union[TensorWrappedPhiTensorPointer, MPCTensor, int, float, np.ndarray],
    ) -> Union[TensorWrappedPhiTensorPointer, MPCTensor]:
        """Apply the "gt" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedPhiTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedPhiTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedPhiTensorPointer._apply_op(self, other, "__gt__")

    def __ge__(
        self,
        other: Union[TensorWrappedPhiTensorPointer, MPCTensor, int, float, np.ndarray],
    ) -> Union[TensorWrappedPhiTensorPointer, MPCTensor]:
        """Apply the "ge" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedPhiTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedPhiTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedPhiTensorPointer._apply_op(self, other, "__ge__")

    def __le__(
        self,
        other: Union[TensorWrappedPhiTensorPointer, MPCTensor, int, float, np.ndarray],
    ) -> Union[TensorWrappedPhiTensorPointer, MPCTensor]:
        """Apply the "le" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedPhiTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedPhiTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedPhiTensorPointer._apply_op(self, other, "__le__")

    def __eq__(  # type: ignore
        self,
        other: Union[TensorWrappedPhiTensorPointer, MPCTensor, int, float, np.ndarray],
    ) -> Union[TensorWrappedPhiTensorPointer, MPCTensor]:
        """Apply the "eq" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedPhiTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedPhiTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedPhiTensorPointer._apply_op(self, other, "__eq__")

    def __ne__(  # type: ignore
        self,
        other: Union[TensorWrappedPhiTensorPointer, MPCTensor, int, float, np.ndarray],
    ) -> Union[TensorWrappedPhiTensorPointer, MPCTensor]:
        """Apply the "ne" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedPhiTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedPhiTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedPhiTensorPointer._apply_op(self, other, "__ne__")

    def __lshift__(
        self,
        other: Union[TensorWrappedPhiTensorPointer, MPCTensor, int, float, np.ndarray],
    ) -> Union[TensorWrappedPhiTensorPointer, MPCTensor]:
        """Apply the "lshift" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedPhiTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedPhiTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedPhiTensorPointer._apply_op(self, other, "__lshift__")

    def __rshift__(
        self,
        other: Union[TensorWrappedPhiTensorPointer, MPCTensor, int, float, np.ndarray],
    ) -> Union[TensorWrappedPhiTensorPointer, MPCTensor]:
        """Apply the "rshift" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedPhiTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedPhiTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedPhiTensorPointer._apply_op(self, other, "__rshift__")

    def __xor__(
        self,
        other: Union[TensorWrappedPhiTensorPointer, MPCTensor, int, float, np.ndarray],
    ) -> Union[TensorWrappedPhiTensorPointer, MPCTensor]:
        """Apply the "xor" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedPhiTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedPhiTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedPhiTensorPointer._apply_op(self, other, "__xor__")

    def concatenate(
        self,
        other: TensorWrappedPhiTensorPointer,
        *args: Any,
        **kwargs: Any,
    ) -> MPCTensor:
        """Apply the "concatenate" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedPhiTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.


        Returns:
            Union[TensorWrappedPhiTensorPointer,MPCTensor] : Result of the operation.
        """
        if not isinstance(other, TensorWrappedPhiTensorPointer):
            raise ValueError(
                f"Concatenate works only for TensorWrappedPhiTensorPointer got type: {type(other)}"
            )

        if self.client != other.client:

            parties = [self.client, other.client]

            self_mpc = MPCTensor(secret=self, shape=self.public_shape, parties=parties)
            other_mpc = MPCTensor(
                secret=other, shape=other.public_shape, parties=parties
            )

            return self_mpc.concatenate(other_mpc, *args, **kwargs)

        else:
            raise ValueError(
                "Concatenate method currently works only between two different clients."
            )

    def __pos__(self) -> TensorWrappedPhiTensorPointer:
        """Apply the __pos__ (+) operator  on self.

        Returns:
            Union[TensorWrappedPhiTensorPointer] : Result of the operation.
        """
        return self._apply_self_tensor_op(op_str="__pos__")

    def __truediv__(
        self,
        other: Union[TensorWrappedPhiTensorPointer, MPCTensor, int, float, np.ndarray],
    ) -> Union[TensorWrappedPhiTensorPointer, MPCTensor]:
        """Apply the "truediv" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedPhiTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedPhiTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedPhiTensorPointer._apply_op(self, other, "__truediv__")

    def __floordiv__(
        self,
        other: Union[TensorWrappedPhiTensorPointer, MPCTensor, int, float, np.ndarray],
    ) -> Union[TensorWrappedPhiTensorPointer, MPCTensor]:
        """Apply the "floordiv" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedPhiTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedPhiTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedPhiTensorPointer._apply_op(self, other, "__floordiv__")

    def __mod__(
        self,
        other: Union[TensorWrappedPhiTensorPointer, MPCTensor, int, float, np.ndarray],
    ) -> Union[
        TensorWrappedPhiTensorPointer, TensorWrappedGammaTensorPointer, MPCTensor
    ]:
        """Apply the "mod" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedPhiTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedPhiTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedPhiTensorPointer._apply_op(self, other, "__mod__")

    def __and__(
        self,
        other: Union[TensorWrappedPhiTensorPointer, MPCTensor, int, float, np.ndarray],
    ) -> Union[
        TensorWrappedPhiTensorPointer, TensorWrappedGammaTensorPointer, MPCTensor
    ]:
        """Apply the "and" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedPhiTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedPhiTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedPhiTensorPointer._apply_op(self, other, "__and__")

    def __or__(
        self,
        other: Union[TensorWrappedPhiTensorPointer, MPCTensor, int, float, np.ndarray],
    ) -> Union[
        TensorWrappedPhiTensorPointer, TensorWrappedGammaTensorPointer, MPCTensor
    ]:
        """Apply the "or" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedPhiTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedPhiTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedPhiTensorPointer._apply_op(self, other, "__or__")

    def __divmod__(
        self,
        other: Union[TensorWrappedPhiTensorPointer, MPCTensor, int, float, np.ndarray],
    ) -> Tuple[
        Union[
            TensorWrappedPhiTensorPointer, TensorWrappedGammaTensorPointer, MPCTensor
        ],
        Union[
            TensorWrappedPhiTensorPointer, TensorWrappedGammaTensorPointer, MPCTensor
        ],
    ]:
        """Apply the "divmod" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedPhiTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedPhiTensorPointer,MPCTensor] : Result of the operation.
        """
        return self.divmod(other)

    def divmod(
        self,
        other: Union[TensorWrappedPhiTensorPointer, MPCTensor, int, float, np.ndarray],
    ) -> Tuple[
        Union[
            TensorWrappedPhiTensorPointer, TensorWrappedGammaTensorPointer, MPCTensor
        ],
        Union[
            TensorWrappedPhiTensorPointer, TensorWrappedGammaTensorPointer, MPCTensor
        ],
    ]:
        """Apply the "divmod" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedPhiTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedPhiTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedPhiTensorPointer._apply_op(
            self, other, "__floordiv__"
        ), TensorWrappedPhiTensorPointer._apply_op(self, other, "__mod__")

    def sum(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Union[
        TensorWrappedPhiTensorPointer, MPCTensor, TensorWrappedGammaTensorPointer
    ]:
        """Apply the "sum" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedPhiTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedPhiTensorPointer,MPCTensor] : Result of the operation.
        """
        return self._apply_self_tensor_op("sum", *args, **kwargs)

    def argmax(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Union[
        TensorWrappedPhiTensorPointer, MPCTensor, TensorWrappedGammaTensorPointer
    ]:
        """Apply the "argmax" operation on self

        Args:
            y (Union[TensorWrappedPhiTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedPhiTensorPointer,MPCTensor] : Result of the operation.
        """
        return self._apply_self_tensor_op("argmax", *args, **kwargs)

    def __abs__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Union[
        TensorWrappedPhiTensorPointer, MPCTensor, TensorWrappedGammaTensorPointer
    ]:
        """Apply the "absolute" operation on self

        Args:
            y (Union[TensorWrappedPhiTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedPhiTensorPointer,MPCTensor] : Result of the operation.
        """
        return self._apply_self_tensor_op("__abs__", *args, **kwargs)

    def all(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Union[
        TensorWrappedPhiTensorPointer, MPCTensor, TensorWrappedGammaTensorPointer
    ]:
        """Apply the "all" operation on self

        Args:
            y (Union[TensorWrappedPhiTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedPhiTensorPointer,MPCTensor] : Result of the operation.
        """
        return self._apply_self_tensor_op("all", *args, **kwargs)

    def any(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Union[
        TensorWrappedPhiTensorPointer, MPCTensor, TensorWrappedGammaTensorPointer
    ]:
        """Apply the "any" operation on self

        Args:
            y (Union[TensorWrappedPhiTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedPhiTensorPointer,MPCTensor] : Result of the operation.
        """
        return self._apply_self_tensor_op("any", *args, **kwargs)

    def argmin(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Union[
        TensorWrappedPhiTensorPointer, MPCTensor, TensorWrappedGammaTensorPointer
    ]:
        """Apply the "argmin" operation on self

        Args:
            y (Union[TensorWrappedPhiTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedPhiTensorPointer,MPCTensor] : Result of the operation.
        """
        return self._apply_self_tensor_op("argmin", *args, **kwargs)

    def ptp(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Union[
        TensorWrappedPhiTensorPointer, MPCTensor, TensorWrappedGammaTensorPointer
    ]:
        """Apply the "ptp" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedPhiTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedPhiTensorPointer,MPCTensor] : Result of the operation.
        """
        return self._apply_self_tensor_op("ptp", *args, **kwargs)

    def __getitem__(
        self, key: Union[int, bool, slice]
    ) -> TensorWrappedPhiTensorPointer:
        """Apply the getitem operation on "self"
        Args:
            y (Union[int,bool,slice]) : second operand.

        Returns:
            Union[TensorWrappedPhiTensorPointer,MPCTensor] : Result of the operation.
        """
        return self._apply_self_tensor_op("__getitem__", key)

    def zeros_like(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> TensorWrappedPhiTensorPointer:
        """Apply the "zeros_like" operation on "self"

        Args:
            y (Union[TensorWrappedPhiTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedPhiTensorPointer,MPCTensor] : Result of the operation.
        """
        return self._apply_self_tensor_op("zeros_like", *args, **kwargs)

    def ones_like(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> TensorWrappedPhiTensorPointer:
        """Apply the "ones_like" operation on "self"

        Args:
            y (Union[TensorWrappedPhiTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedPhiTensorPointer,MPCTensor] : Result of the operation.
        """
        return self._apply_self_tensor_op("ones_like", *args, **kwargs)

    def repeat(self, *args: Any, **kwargs: Any) -> TensorWrappedPhiTensorPointer:
        """Apply the "repeat" operation

        Args:
            y (Union[TensorWrappedPhiTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedPhiTensorPointer,MPCTensor] : Result of the operation.
        """
        return self._apply_self_tensor_op("repeat", *args, **kwargs)

    def var(self, *args: Any, **kwargs: Any) -> TensorWrappedPhiTensorPointer:
        """
        Compute the variance along the specified axis of the array elements, a measure of the spread of a distribution.
        The variance is computed for the flattened array by default, otherwise over the specified axis.

        Parameters

            axis: None or int or tuple of ints, optional
                Axis or axes along which the variance is computed.
                The default is to compute the variance of the flattened array.
                If this is a tuple of ints, a variance is performed over multiple axes, instead of a single axis or all
                the axes as before.

            ddof: int, optional
                “Delta Degrees of Freedom”: the divisor used in the calculation is N - ddof, where N represents the
                number of elements. By default ddof is zero.

            keepdims: bool, optional
                If this is set to True, the axes which are reduced are left in the result as dimensions with size one.
                With this option, the result will broadcast correctly against the input array.
                If the default value is passed, then keepdims will not be passed through to the var method of
                sub-classes of ndarray, however any non-default value will be. If the sub-class’ method does not
                implement keepdims any exceptions will be raised.

            where: array_like of bool, optional
                Elements to include in the variance. See reduce for details.
        """
        return self._apply_self_tensor_op("var", *args, **kwargs)

    def cumsum(self, *args: Any, **kwargs: Any) -> TensorWrappedPhiTensorPointer:
        """
        Return the cumulative sum of the elements along a given axis.

        Parameters
            axis: int, optional
                Axis along which the cumulative sum is computed. The default (None) is to compute the cumsum over the
                flattened array.
        Returns
            cumsum_along_axis: PhiTensor
                A new array holding the result is returned. The result has the same size as input, and the same shape as
                 a if axis is not None or a is 1-d.
        """
        return self._apply_self_tensor_op("cumsum", *args, **kwargs)

    def prod(self, *args: Any, **kwargs: Any) -> TensorWrappedPhiTensorPointer:
        """
        Return the product of array elements over a given axis.

        Parameters
            axis: None or int or tuple of ints, optional
                Axis or axes along which a product is performed.
                The default, axis=None, will calculate the product of all the elements in the input array.
                If axis is negative it counts from the last to the first axis.

                If axis is a tuple of ints, a product is performed on all of the axes specified in the tuple instead of
                a single axis or all the axes as before.

            keepdims: bool, optional
                If this is set to True, the axes which are reduced are left in the result as dimensions with size one.
                With this option, the result will broadcast correctly against the input array.

                If the default value is passed, then keepdims will not be passed through to the prod method of
                sub-classes of ndarray, however any non-default value will be. If the sub-class’ method does not
                implement keepdims any exceptions will be raised.

            initial: scalar, optional
                The starting value for this product. See reduce for details.

            where: array_like of bool, optional
                Elements to include in the product. See reduce for details.
        """
        return self._apply_self_tensor_op("prod", *args, **kwargs)

    def __pow__(self, *args: Any, **kwargs: Any) -> TensorWrappedPhiTensorPointer:
        """
        First array elements raised to powers from second array, element-wise.

        Raise each base in x1 to the positionally-corresponding power in x2.
        x1 and x2 must be broadcastable to the same shape.
        An integer type raised to a negative integer power will raise a ValueError.
        Negative values raised to a non-integral value will return nan.

        Parameters
            x2: array_like

                The exponents. If self.shape != x2.shape, they must be broadcastable to a common shape.

            where: array_like, optional

                This condition is broadcast over the input. At locations where the condition is True, the out array will
                 be set to the ufunc result.
                 Elsewhere, the out array will retain its original value.

            **kwargs
                For other keyword-only arguments, see the ufunc docs.

        Returns
            y: PhiTensorPointer
                The bases in the tensor raised to the exponents in x2. This is a scalar if both self and x2 are scalars.
        """
        return self._apply_self_tensor_op("__pow__", *args, **kwargs)

    def mean(self, *args: Any, **kwargs: Any) -> TensorWrappedPhiTensorPointer:
        """
        Compute the arithmetic mean along the specified axis.

        Returns the average of the array elements. The average is taken over the flattened array by default, otherwise
        over the specified axis.

        Parameters
            axis: None or int or tuple of ints, optional
                Axis or axes along which the means are computed. The default is to compute the mean of the flattened
                array.
        """
        return self._apply_self_tensor_op("mean", *args, **kwargs)

    def std(self, *args: Any, **kwargs: Any) -> TensorWrappedPhiTensorPointer:
        """
        Compute the standard deviation along the specified axis.
        Returns the standard deviation, a measure of the spread of a distribution, of the array elements.
        The standard deviation is computed for the flattened array by default, otherwise over the specified axis.

        Parameters
            axis: None or int or tuple of ints, optional
                Axis or axes along which the standard deviation is computed.
                The default is to compute the standard deviation of the flattened array.
                If this is a tuple of ints, a standard deviation is performed over multiple axes, instead of a single
                axis or all the axes as before.

            out: ndarray, optional
                Alternative output array in which to place the result. It must have the same shape as the expected
                output but the type (of the calculated values) will be cast if necessary.

            ddof: int, optional
                ddof = Delta Degrees of Freedom. By default ddof is zero.
                The divisor used in calculations is N - ddof, where N represents the number of elements.

            keepdims: bool, optional
                If this is set to True, the axes which are reduced are left in the result as dimensions with size one.
                With this option, the result will broadcast correctly against the input array.

                If the default value is passed, then keepdims will not be passed through to the std method of
                sub-classes of ndarray, however any non-default value will be. If the sub-class’ method does not
                implement keepdims any exceptions will be raised.

            where: array_like of bool, optional
                Elements to include in the standard deviation. See reduce for details.

        Returns

            standard_deviation: PhiTensor
        """

        attr_path_and_name = "syft.core.tensor.tensor.Tensor.std"
        result: TensorWrappedPhiTensorPointer
        data_subjects = np.array(self.data_subjects.std(*args, **kwargs))
        result = TensorWrappedPhiTensorPointer(
            data_subjects=self.data_subjects,
            min_vals=lazyrepeatarray(data=0, shape=data_subjects.shape),
            max_vals=lazyrepeatarray(
                data=(self.max_vals.data - self.min_vals.data) / 2,
                shape=data_subjects.shape,
            ),
            client=self.client,
        )

        # QUESTION can the id_at_location be None?
        result_id_at_location = getattr(result, "id_at_location", None)

        if result_id_at_location is not None:
            # first downcast anything primitive which is not already PyPrimitive
            (
                downcast_args,
                downcast_kwargs,
            ) = lib.python.util.downcast_args_and_kwargs(args=args, kwargs=kwargs)

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
            args=[],
            kwargs={},
        )

        result.public_shape = data_subjects.shape
        result.public_dtype = self.public_dtype
        return result

    def cumprod(self, *args: Any, **kwargs: Any) -> TensorWrappedPhiTensorPointer:
        """
        Return the cumulative product of the elements along a given axis.

        Parameters
            axis: int, optional
                Axis along which the cumulative product is computed. The default (None) is to compute the cumprod over
                the flattened array.
        Returns
            cumprod_along_axis: PhiTensor
                A new array holding the result is returned. The result has the same size as input, and the same shape as
                 a if axis is not None or a is 1-d.
        """
        return self._apply_self_tensor_op("cumprod", *args, **kwargs)

    def take(self, *args: Any, **kwargs: Any) -> TensorWrappedPhiTensorPointer:
        """
        Take elements from an array along an axis.

        When axis is not None, this function does the same thing as “fancy” indexing (indexing arrays using arrays);
        however, it can be easier to use if you need elements along a given axis.
        A call such as np.take(arr, indices, axis=3) is equivalent to arr[:,:,:,indices,...].

        Explained without fancy indexing, this is equivalent to the following use of ndindex, \
        which sets each of ii, jj, and kk to a tuple of indices:

            Ni, Nk = a.shape[:axis], a.shape[axis+1:]
            Nj = indices.shape
            for ii in ndindex(Ni):
                for jj in ndindex(Nj):
                    for kk in ndindex(Nk):
                        out[ii + jj + kk] = a[ii + (indices[jj],) + kk]

        Parameters
            indices: array_like (Nj…)
                The indices of the values to extract.

            axis: int, optional
                The axis over which to select values. By default, the flattened input array is used.

            mode: {‘raise’, ‘wrap’, ‘clip’}, optional
                Specifies how out-of-bounds indices will behave.

                * ‘raise’ – raise an error (default)

                * ‘wrap’ – wrap around

                * ‘clip’ – clip to the range

                ‘clip’ mode means that all indices that are too large are replaced by the index
                that addresses the last element along that axis.
                Note that this disables indexing with negative numbers.

        Returns
            out: PhiTensor
                The returned array has the same type as a.
        """
        return self._apply_self_tensor_op("take", *args, **kwargs)

    def flatten(self, *args: Any, **kwargs: Any) -> TensorWrappedPhiTensorPointer:
        """
        Return a copy of the array collapsed into one dimension.

        Parameters
            order: {‘C’, ‘F’, ‘A’, ‘K’}, optional
                ‘C’ means to flatten in row-major (C-style) order.
                ‘F’ means to flatten in column-major (Fortran- style) order.
                ‘A’ means to flatten in column-major order if a is Fortran contiguous in memory,
                   row-major order otherwise.
                ‘K’ means to flatten a in the order the elements occur in memory.
                The default is ‘C’.

        Returns
            y: PhiTensor
                A copy of the input array, flattened to one dimension.
        """
        return self._apply_self_tensor_op("flatten", *args, **kwargs)

    def ravel(self, *args: Any, **kwargs: Any) -> TensorWrappedPhiTensorPointer:
        """
        Return a contiguous flattened array.

        A 1-D array, containing the elements of the input, is returned. A copy is made only if needed.

        As of NumPy 1.10, the returned array will have the same type as the input array.
        (for example, a masked array will be returned for a masked array input)
        Parameters
            order: {‘C’,’F’, ‘A’, ‘K’}, optional
                The elements of a are read using this index order.
                ‘C’ means to index the elements in row-major,
                C-style order, with the last axis index changing fastest, back to the first axis index
                changing slowest.
                ‘F’ means to index the elements in column-major, Fortran-style order, with
                the first index changing fastest,
                and the last index changing slowest.
                Note that the ‘C’ and ‘F’ options take no account of the memory layout of the underlying array,
                and only refer to the order of axis indexing.
                ‘A’ means to read the elements in Fortran-like index order if a is Fortran contiguous in memory,
                C-like order otherwise.
                ‘K’ means to read the elements in the order they occur in memory, except for reversing the data
                when strides are negative.
                By default, ‘C’ index order is used.

        Returns:
            y: PhiTensor
                y is an array of the same subtype as a, with shape (a.size,).
                Note that matrices are special cased for backward compatibility, if a is a matrix,
                then y is a 1-D ndarray.
        """
        return self._apply_self_tensor_op("ravel", *args, **kwargs)

    def compress(self, *args: Any, **kwargs: Any) -> TensorWrappedPhiTensorPointer:
        """
        Return selected slices of an array along given axis.

        When working along a given axis, a slice along that axis is returned in output
        for each index where condition evaluates to True. When working on a 1-D array,
        compress is equivalent to extract.

        Parameters
            condition: 1-D array of bools
                Array that selects which entries to return. If len(condition) is less than the
                 size of a along the given axis,
                then output is truncated to the length of the condition array.

            axis: int, optional
                Axis along which to take slices. If None (default), work on the flattened array.

        Returns:
            compressed_array: PhiTensor
                A copy of a without the slices along axis for which condition is false.
        """
        return self._apply_self_tensor_op("compress", *args, **kwargs)

    def squeeze(self, *args: Any, **kwargs: Any) -> TensorWrappedPhiTensorPointer:
        """
        Remove axes of length one from a.

        Parameters
            axis: None or int or tuple of ints, optional
                Selects a subset of the entries of length one in the shape.
                If an axis is selected with shape entry greater than one, an error is raised.

        Returns:
            squeezed: PhiTensor
                The input array, but with all or a subset of the dimensions of length 1 removed.
                This is always a itself or a view into a.
                Note that if all axes are squeezed, the result is a 0d array and not a scalar.
        """
        return self._apply_self_tensor_op("squeeze", *args, **kwargs)

    def trace(self, *args: Any, **kwargs: Any) -> TensorWrappedPhiTensorPointer:
        """
        Return the sum along diagonals of the array.

        If a is 2-D, the sum along its diagonal with the given offset is returned,
        i.e., the sum of elements a[i,i+offset] for all i.

        If a has more than two dimensions, then the axes specified by axis1 and axis2 are used to determine the 2-D
        sub-arrays whose traces are returned. The shape of the resulting array is the same as that of a with axis1 and
        axis2 removed.

        Parameters

            offset: int, optional
                Offset of the diagonal from the main diagonal. Can be both positive and negative. Defaults to 0.

            axis1, axis2: int, optional
                Axes to be used as the first and second axis of the 2-D sub-arrays from which the diagonals should be
                taken. Defaults are the first two axes of a.

        Returns

            Union[TensorWrappedPhiTensorPointer,MPCTensor] : Result of the operation.
                If a is 2-D, the sum along the diagonal is returned.
                If a has larger dimensions, then an array of sums along diagonals is returned.

        """
        return self._apply_self_tensor_op("trace", *args, **kwargs)

    def min(self, *args: Any, **kwargs: Any) -> TensorWrappedPhiTensorPointer:
        """
        Return the minimum of an array or minimum along an axis.

        Parameters
            axis: None or int or tuple of ints, optional
                Axis or axes along which to operate. By default, flattened input is used.
                If this is a tuple of ints, the minimum is selected over multiple axes,
                instead of a single axis or all the axes as before.

        Returns
            a_min: PhiTensor
                Minimum of a.
                If axis is None, the result is a scalar value.
                If axis is given, the result is an array of dimension a.ndim - 1.
        """
        return self._apply_self_tensor_op("min", *args, **kwargs)

    def max(self, *args: Any, **kwargs: Any) -> TensorWrappedPhiTensorPointer:
        """
        Return the maximum of an array or along an axis.

        Parameters
            axis: None or int or tuple of ints, optional
                Axis or axes along which to operate. By default, flattened input is used.
                If this is a tuple of ints, the minimum is selected over multiple axes,
                instead of a single axis or all the axes as before.

        Returns
            a_max: PhiTensor
                Maximum of a.
                If axis is None, the result is a scalar value.
                If axis is given, the result is an array of dimension a.ndim - 1.
        """
        return self._apply_self_tensor_op("max", *args, **kwargs)

    def sort(self, *args: Any, **kwargs: Any) -> TensorWrappedPhiTensorPointer:
        """
        Return a sorted copy of an array.

        Parameters

            a: array_like
                Array to be sorted.

            axis: int or None, optional
                Axis along which to sort. If None, the array is flattened before sorting.
                The default is -1, which sorts along the last axis.

            kind{‘quicksort’, ‘mergesort’, ‘heapsort’, ‘stable’}, optional
                Sorting algorithm. The default is ‘quicksort’.
                Note that both ‘stable’ and ‘mergesort’ use timsort or radix sort under the covers and, in general,
                the actual implementation will vary with data type. The ‘mergesort’ option is retained for backwards
                compatibility.

                Changed in version 1.15.0.: The ‘stable’ option was added.

            order: str or list of str, optional
                When a is an array with fields defined, this argument specifies which fields to compare first, second,
                etc. A single field can be specified as a string, and not all fields need be specified, but unspecified
                 fields will still be used, in the order in which they come up in the dtype, to break ties.

        Please see docs here: https://numpy.org/doc/stable/reference/generated/numpy.sort.html
        """
        return self._apply_self_tensor_op("sort", *args, **kwargs)

    def argsort(self, *args: Any, **kwargs: Any) -> TensorWrappedPhiTensorPointer:
        """
        Returns the indices that would sort an array.

        Perform an indirect sort along the given axis using the algorithm specified by the kind keyword.
        It returns an array of indices of the same shape as a that index data along the given axis in sorted order.

        Parameters
            axis: int or None, optional
                Axis along which to sort. The default is -1 (the last axis). If None, the flattened array is used.
            kind: {‘quicksort’, ‘mergesort’, ‘heapsort’, ‘stable’}, optional
                Sorting algorithm. The default is ‘quicksort’. Note that both ‘stable’ and ‘mergesort’ use timsort
                under the covers and, in general, the actual implementation will vary with data type. The ‘mergesort’
                option is retained for backwards compatibility.
            order: str or list of str, optional
                When a is an array with fields defined, this argument specifies which fields to compare 1st, 2nd, etc.
                A single field can be specified as a string, and not all fields need be specified, but unspecified
                fields will still be used, in the order in which they come up in the dtype, to break ties.

        Returns
            index_array: ndarray, int
                Array of indices that sort a along the specified axis. If a is one-dimensional, a[index_array] yields a
                sorted a. More generally, np.take_along_axis(a, index_array, axis=axis) always yields the sorted a,
                irrespective of dimensionality.
        """
        return self._apply_self_tensor_op("argsort", *args, **kwargs)

    def diagonal(self, *args: Any, **kwargs: Any) -> TensorWrappedPhiTensorPointer:
        """
        Return the sum along diagonals of the array.

        Return specified diagonals.
        If a is 2-D, returns the diagonal of a with the given offset, i.e., the collection of
        elements of the form a[i, i+offset].

        If a has more than two dimensions, then the axes specified by axis1 and axis are used
        to determine the 2-D sub-array whose diagonal is returned.  The shape of the resulting
        array can be determined by removing axis1 and axis2 and appending an index to the right
        equal to the size of the resulting diagonals.

        Parameters

            offset: int, optional
                Offset of the diagonal from the main diagonal.  Can be positive or negative.
                Defaults to main diagonal (0).
            axis1, axis2: int, optional
                Axis to be used as the first axis of the 2-D sub-arrays from which the diagonals
                should be taken. Defaults are the first two axes of a.

        Returns
            array_of_diagonals : Union[TensorWrappedPhiTensorPointer,MPCTensor]
                If a is 2-D, then a 1-D array containing the diagonal and of the same type as a
                is returned unless a is a matrix, in which casea 1-D array rather than a (2-D)
                matrix is returned in order to maintain backward compatibility.

                If a.ndim > 2, then the dimensions specified by axis1 and axis2 are removed,
                and a new axis inserted at the end corresponding to the diagonal.
        """
        return self._apply_self_tensor_op("diagonal", *args, **kwargs)

    def clip(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> TensorWrappedPhiTensorPointer:
        """
        Clip (limit) the values in an array.

        Parameters
            a : array_like
                Array containing elements to clip.
            a_min, a_max : array_like or None
                Minimum and maximum value. If None, clipping is not performed on
                the corresponding edge. Only one of a_min and a_max may be
                None. Both are broadcast against a.
        Returns:
            Union[TensorWrappedPhiTensorPointer,MPCTensor] : Result of the operation.
        """
        return self._apply_self_tensor_op("clip", *args, **kwargs)

    def choose(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> TensorWrappedPhiTensorPointer:
        """
        Construct an array from an index array and a list of arrays to choose from.

        First of all, if confused or uncertain, definitely look at the Examples - in its full generality,
        this function is less simple than it might seem from the following code description
        (below ndi = numpy.lib.index_tricks):

        np.choose(a,c) == np.array([c[a[I]][I] for I in ndi.ndindex(a.shape)]).

        But this omits some subtleties. Here is a fully general summary:

        Given an “index” array (a) of integers and a sequence of n arrays (choices), a and each choice array are first
        broadcast, as necessary, to arrays of a common shape; calling these Ba and Bchoices[i], i = 0,…,n-1 we have that
         necessarily, Ba.shape == Bchoices[i].shape for each i. Then, a new array with shape Ba.shape is created
         as follows:

            if mode='raise' (the default), then, first of all, each element of a (and thus Ba) must be in the range
            [0, n-1]; now, suppose that i (in that range) is the value at the (j0, j1, ..., jm) position in Ba -
            then the value at the same position in the new array is the value in Bchoices[i] at that same position;

            if mode='wrap', values in a (and thus Ba) may be any (signed) integer; modular arithmetic is used to map
            integers outside the range [0, n-1] back into that range; and then the new array is constructed as above;

            if mode='clip', values in a (and thus Ba) may be any (signed) integer; negative integers are mapped to 0;
            values greater than n-1 are mapped to n-1; and then the new array is constructed as above.

        Parameters

            choices: sequence of arrays
                Choice arrays. a and all of the choices must be broadcastable to the same shape. If choices is itself an
                 array (not recommended), then its outermost dimension (i.e., the one corresponding to choices.shape[0])
                  is taken as defining the “sequence”.

            out: array, optional
                If provided, the result will be inserted into this array. It should be of the appropriate shape and
                dtype. Note that out is always buffered if mode='raise'; use other modes for better performance.

            mode{‘raise’ (default), ‘wrap’, ‘clip’}, optional
                Specifies how indices outside [0, n-1] will be treated:
                        ‘raise’ : an exception is raised
                        ‘wrap’ : value becomes value mod n
                        ‘clip’ : values < 0 are mapped to 0, values > n-1 are mapped to n-1

        Returns
            merged_array: PhiTensor
                The merged result.

        Raises
            ValueError: shape mismatch
                If a and each choice array are not all broadcastable to the same shape.

        """
        return self._apply_self_tensor_op("choose", *args, **kwargs)

    @property
    def T(self) -> TensorWrappedPhiTensorPointer:
        # We always maintain a Tensor hierarchy Tensor ---> PT--> Actual Data
        attr_path_and_name = "syft.core.tensor.tensor.Tensor.T"

        result = TensorWrappedPhiTensorPointer(
            data_subjects=self.data_subjects,
            min_vals=self.min_vals.transpose(),
            max_vals=self.max_vals.transpose(),
            client=self.client,
        )

        # QUESTION can the id_at_location be None?
        result_id_at_location = getattr(result, "id_at_location", None)

        if result_id_at_location is not None:
            # first downcast anything primitive which is not already PyPrimitive
            (
                downcast_args,
                downcast_kwargs,
            ) = lib.python.util.downcast_args_and_kwargs(args=[], kwargs={})

            # then we convert anything which isnt a pointer into a pointer
            pointer_args, pointer_kwargs = pointerize_args_and_kwargs(
                args=downcast_args,
                kwargs=downcast_kwargs,
                client=self.client,
                gc_enabled=False,
            )

            cmd = GetOrSetPropertyAction(
                path=attr_path_and_name,
                id_at_location=result_id_at_location,
                address=self.client.address,
                _self=self,
                args=pointer_args,
                kwargs=pointer_kwargs,
                action=PropertyActions.GET,
                map_to_dyn=False,
            )
            self.client.send_immediate_msg_without_reply(msg=cmd)

        inherit_tags(
            attr_path_and_name=attr_path_and_name,
            result=result,
            self_obj=self,
            args=[],
            kwargs={},
        )

        result_public_shape = np.empty(self.public_shape).T.shape

        result.public_shape = result_public_shape
        result.public_dtype = self.public_dtype

        return result

    def to_local_object_without_private_data_child(self) -> PhiTensor:
        """Convert this pointer into a partial version of the PhiTensor but without
        any of the private data therein."""
        # relative
        from ..tensor import Tensor

        public_shape = getattr(self, "public_shape", None)
        public_dtype = getattr(self, "public_dtype", None)
        return Tensor(
            child=PhiTensor(
                child=FixedPrecisionTensor(value=np.empty(self.data_subjects.shape)),
                data_subjects=self.data_subjects,
                min_vals=self.min_vals,  # type: ignore
                max_vals=self.max_vals,  # type: ignore
            ),
            public_shape=public_shape,
            public_dtype=public_dtype,
        )

    def transpose(self, *args: Any, **kwargs: Any) -> TensorWrappedPhiTensorPointer:
        """
        Reverse or permute the axes of an array; returns the modified array.

        Returns
            p: ndarray
                array with its axes permuted. A view is returned whenever possible.
        """
        return self._apply_self_tensor_op("transpose", *args, **kwargs)

    def resize(self, *args: Any, **kwargs: Any) -> TensorWrappedPhiTensorPointer:
        """
        Return a new array with the specified shape.

        Parameters
            new_shape: int or tuple of int
                Shape of resized array.

        Returns
            reshaped_array: ndarray
                The new array is formed from the data in the old array,
                repeated if necessary to fill out the required number of elements.
                The data are repeated iterating over the array in C-order.

        """
        return self._apply_self_tensor_op("resize", *args, **kwargs)

    def reshape(self, *args: Any, **kwargs: Any) -> TensorWrappedPhiTensorPointer:
        """
        Gives a new shape to an array without changing its data.

        Parameters
            new_shape: int or tuple of int
                The new shape should be compatible with the original shape. If an integer, then the result will
                be a 1-D array of that length. One shape dimension can be -1. In this case,
                the value is inferred from the length of the array and remaining dimensions.

        Returns
            reshaped_array: ndarray
                This will be a new view object if possible; otherwise, it will be a copy.
                Note there is no guarantee of the memory layout (C- or Fortran- contiguous) of the returned array.
        """
        return self._apply_self_tensor_op("reshape", *args, **kwargs)


@implements(TensorWrappedPhiTensorPointer, np.zeros_like)
def zeros_like(
    tensor: TensorWrappedPhiTensorPointer,
    *args: Any,
    **kwargs: Any,
) -> TensorWrappedPhiTensorPointer:
    return tensor.zeros_like(*args, **kwargs)


@implements(TensorWrappedPhiTensorPointer, np.ones_like)
def ones_like(
    tensor: TensorWrappedPhiTensorPointer,
    *args: Any,
    **kwargs: Any,
) -> TensorWrappedPhiTensorPointer:
    return tensor.ones_like(*args, **kwargs)


@serializable(capnp_bytes=True)
class PhiTensor(PassthroughTensor, ADPTensor):
    PointerClassOverride = TensorWrappedPhiTensorPointer
    # __attr_allowlist__ = ["child", "min_vals", "max_vals", "data_subjects"]
    __slots__ = (
        "child",
        "min_vals",
        "max_vals",
        "data_subjects",
    )

    def __init__(
        self,
        child: Union[Sequence, NDArray],
        data_subjects: Union[DataSubjectArray, NDArray],
        min_vals: Union[np.ndarray, lazyrepeatarray],
        max_vals: Union[np.ndarray, lazyrepeatarray],
    ) -> None:
        # self.data_subjects: Union[DataSubjectList, np.ndarray]
        # child = the actual private data
        super().__init__(child)

        # lazyrepeatarray matching the shape of child
        if not isinstance(min_vals, lazyrepeatarray):
            min_vals = lazyrepeatarray(data=min_vals, shape=child.shape)  # type: ignore
        if not isinstance(max_vals, lazyrepeatarray):
            max_vals = lazyrepeatarray(data=max_vals, shape=child.shape)  # type: ignore
        self.min_vals = min_vals
        self.max_vals = max_vals

        numpy_data_subjects: np.ndarray = DataSubjectArray.from_objs(data_subjects)
        self.data_subjects = numpy_data_subjects
        if numpy_data_subjects.shape != self.shape:
            raise ValueError(
                f"DataSubjects shape: {numpy_data_subjects.shape} should match data shape: {self.shape}"
            )

    @property
    def proxy_public_kwargs(self) -> Dict[str, Any]:
        return {
            "min_vals": self.min_vals,
            "max_vals": self.max_vals,
            "data_subjects": self.data_subjects,
        }

    # def init_pointer(
    #     self,
    #     client: Any,
    #     id_at_location: Optional[UID] = None,
    #     object_type: str = "",
    #     tags: Optional[List[str]] = None,
    #     description: str = "",
    # ) -> TensorWrappedPhiTensorPointer:
    #     return TensorWrappedPhiTensorPointer(
    #         # Arguments specifically for SEPhiTensor
    #         data_subjects=self.data_subjects,
    #         min_vals=self.min_vals,
    #         max_vals=self.max_vals,
    #         # Arguments required for a Pointer to work
    #         client=client,
    #         id_at_location=id_at_location,
    #         object_type=object_type,
    #         tags=tags,
    #         description=description,
    #     )

    @property
    def gamma(self) -> GammaTensor:
        """Property to cast this tensor into a GammaTensor"""
        return self.create_gamma()

    def copy(self, order: Optional[str] = "K") -> PhiTensor:
        """Return copy of the given object"""

        return PhiTensor(
            child=self.child.copy(order=order),
            min_vals=self.min_vals.copy(order=order),
            max_vals=self.max_vals.copy(order=order),
            data_subjects=self.data_subjects.copy(order=order),
        )

    def take(
        self,
        indices: ArrayLike,
        axis: Optional[int] = None,
        out: Optional[np.ndarray] = None,
        mode: str = "raise",
    ) -> PhiTensor:
        """Take elements from an array along an axis."""
        out_child = self.child.take(indices, axis=axis, mode=mode, out=out)
        return PhiTensor(
            child=out_child,
            min_vals=lazyrepeatarray(data=self.min_vals.data, shape=out_child.shape),
            max_vals=lazyrepeatarray(data=self.max_vals.data, shape=out_child.shape),
            data_subjects=self.data_subjects.take(
                indices, axis=axis, mode=mode, out=out
            ),
        )

    def put(
        self,
        ind: ArrayLike,
        v: ArrayLike,
        mode: str = "raise",
    ) -> PhiTensor:
        """Replaces specified elements of an array with given values.
        The indexing works on the flattened target array. put is roughly equivalent to:
            a.flat[ind] = v
        """
        if self.min_vals.data > min(v) or self.max_vals.data < max(v):
            raise Exception("The v values must be within the data bounds")

        out_child = self.child
        out_child.put(ind, v, mode=mode)
        return PhiTensor(
            child=out_child,
            min_vals=self.min_vals,
            max_vals=self.max_vals,
            data_subjects=self.data_subjects,
        )

    def ptp(
        self,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
    ) -> PhiTensor:
        out_child = self.child.ptp(axis=axis)

        argmin = self.child.argmin(axis=axis)
        argmax = self.child.argmax(axis=axis)

        if axis is None:
            max_indices = np.unravel_index(argmax, shape=self.child.shape)
            min_indices = np.unravel_index(argmin, shape=self.child.shape)
            data_subjects = (
                self.data_subjects[max_indices] - self.data_subjects[min_indices]
            )
        else:
            max_indices = np.array([argmax])
            min_indices = np.array([argmin])
            data_subjects_max = np.squeeze(
                np.take_along_axis(self.data_subjects, max_indices, axis=axis)
            )
            data_subjects_min = np.squeeze(
                np.take_along_axis(self.data_subjects, min_indices, axis=axis)
            )
            data_subjects = data_subjects_max - data_subjects_min

        return PhiTensor(
            child=out_child,
            min_vals=lazyrepeatarray(data=0, shape=out_child.shape),
            max_vals=lazyrepeatarray(
                data=self.max_vals.data - self.min_vals.data, shape=out_child.shape
            ),
            data_subjects=data_subjects,
        )

    def __mod__(self, other: SupportedChainType) -> Union[PhiTensor, GammaTensor]:

        # if the tensor being added is also private
        if isinstance(other, PhiTensor):
            if (self.data_subjects != other.data_subjects).any():
                return self.gamma % other.gamma
            else:
                out_child = self.child % other.child
                return PhiTensor(
                    child=self.child % other.child,
                    data_subjects=self.data_subjects,
                    min_vals=lazyrepeatarray(
                        data=min(0, other.min_vals.data), shape=out_child.shape
                    ),
                    max_vals=lazyrepeatarray(
                        data=max(0, other.max_vals.data), shape=out_child.shape
                    ),
                )

        # if the tensor being added is a public tensor / int / float / etc.
        elif is_acceptable_simple_type(other):

            if isinstance(other, np.ndarray):
                max_vals = lazyrepeatarray(
                    data=max(0, other.max()), shape=self.child.shape
                )
                min_vals = lazyrepeatarray(
                    data=min(0, other.min()), shape=self.child.shape
                )
            else:
                max_vals = lazyrepeatarray(data=max(0, other), shape=self.child.shape)
                min_vals = lazyrepeatarray(data=min(0, other), shape=self.child.shape)

            return PhiTensor(
                child=self.child % other,
                min_vals=min_vals,
                max_vals=max_vals,
                data_subjects=self.data_subjects,
            )

        elif isinstance(other, GammaTensor):
            return self.gamma % other
        else:
            print("Type is unsupported:" + str(type(other)))
            raise NotImplementedError

    def any(
        self,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: Optional[bool] = False,
        where: Optional[ArrayLike] = None,
    ) -> PhiTensor:
        # TODO: properly define data subjects and
        # figure out if it is not a privacy violation to return bool
        if where is None:
            out_child = np.array(self.child.any(axis=axis, keepdims=keepdims))
            new_data_subjects = np.add.reduce(
                self.data_subjects,
                axis=axis,
                keepdims=keepdims,
            )
        else:
            out_child = np.array(
                self.child.any(axis=axis, keepdims=keepdims, where=where)
            )
            new_data_subjects = np.add.reduce(
                self.data_subjects,
                axis=axis,
                keepdims=keepdims,
                initial=DataSubject(),
                where=where,
            )

        return PhiTensor(
            child=out_child,
            min_vals=lazyrepeatarray(data=0, shape=out_child.shape),
            max_vals=lazyrepeatarray(data=1, shape=out_child.shape),
            data_subjects=new_data_subjects,
        )

    def all(
        self,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: Optional[bool] = False,
        where: Optional[ArrayLike] = None,
    ) -> PhiTensor:
        # TODO: properly define data subjects
        if where is None:
            out_child = np.array(self.child.all(axis=axis, keepdims=keepdims))
            new_data_subjects = np.add.reduce(
                self.data_subjects,
                axis=axis,
                keepdims=keepdims,
            )
        else:
            out_child = np.array(
                self.child.all(axis=axis, keepdims=keepdims, where=where)
            )
            new_data_subjects = np.add.reduce(
                self.data_subjects,
                axis=axis,
                keepdims=keepdims,
                initial=DataSubject(),
                where=where,
            )

        return PhiTensor(
            child=out_child,
            min_vals=lazyrepeatarray(data=0, shape=out_child.shape),
            max_vals=lazyrepeatarray(data=1, shape=out_child.shape),
            data_subjects=new_data_subjects,
        )

    def __and__(self, other: SupportedChainType) -> Union[PhiTensor, GammaTensor]:
        # if the tensor being added is also private
        if isinstance(other, PhiTensor):
            if (self.data_subjects != other.data_subjects).any():
                return self.gamma & other.gamma
            else:
                child = self.child & other.child
                other_min, other_max = other.min_vals.data, other.max_vals.data

        # if the tensor being added is a public tensor / int / float / etc.
        elif is_acceptable_simple_type(other):
            child = self.child & other
            if isinstance(other, np.ndarray):
                other_min, other_max = other.min(), other.max()
            else:
                other_min, other_max = other, other

        elif isinstance(other, GammaTensor):
            return self.gamma & other
        else:
            print("Type is unsupported:" + str(type(other)))
            raise NotImplementedError

        # TODO: should modify for a tighter found for and
        _max_vals = int(max(self.max_vals.data.max(), other_max))
        _min = int(min(self.min_vals.data.min(), other_min))
        if self.min_vals.data < 0 and other_min < 0:
            _min_vals = -(2 ** _min.bit_length())
        else:
            _min_vals = min(0, _min)

        return PhiTensor(
            child=child,
            data_subjects=self.data_subjects,
            min_vals=lazyrepeatarray(data=_min_vals, shape=child.shape),
            max_vals=lazyrepeatarray(data=_max_vals, shape=child.shape),
        )

    def __or__(self, other: SupportedChainType) -> Union[PhiTensor, GammaTensor]:
        # if the tensor being added is also private
        if isinstance(other, PhiTensor):
            if (self.data_subjects != other.data_subjects).any():
                return self.gamma | other.gamma
            else:
                child = self.child | other.child
                other_min, other_max = other.min_vals.data, other.max_vals.data

        # if the tensor being added is a public tensor / int / float / etc.
        elif is_acceptable_simple_type(other):
            if isinstance(other, np.ndarray):
                other_min, other_max = other.min(), other.max()
            else:
                other_min, other_max = other, other
            child = self.child | other

        elif isinstance(other, GammaTensor):
            return self.gamma | other
        else:
            print("Type is unsupported:" + str(type(other)))
            raise NotImplementedError

        # TODO: should modify for a tighter found for or
        _max = int(max(self.max_vals.data, other_max))
        _min_vals = min(self.min_vals.data, other_min)
        _max_vals = (2 ** (_max).bit_length()) - 1

        return PhiTensor(
            child=child,
            min_vals=lazyrepeatarray(data=_min_vals, shape=child.shape),
            max_vals=lazyrepeatarray(data=_max_vals, shape=child.shape),
            data_subjects=self.data_subjects,
        )

    def copy_with(self, child: np.ndarray) -> PhiTensor:
        new_tensor = self.copy()
        new_tensor.child = child
        return new_tensor

    def __getitem__(self, item: Union[str, int, slice, PassthroughTensor]) -> PhiTensor:
        if isinstance(item, PassthroughTensor):
            data = self.child[item.child]
            return PhiTensor(
                child=data,
                min_vals=lazyrepeatarray(data=data, shape=data.shape),
                max_vals=lazyrepeatarray(data=data, shape=data.shape),
                data_subjects=self.data_subjects[item.child],
            )
        else:
            data = self.child[item]
            return PhiTensor(
                child=data,
                min_vals=lazyrepeatarray(data=data, shape=data.shape),
                max_vals=lazyrepeatarray(data=data, shape=data.shape),
                data_subjects=self.data_subjects[item],
            )

    def zeros_like(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Union[PhiTensor, GammaTensor]:
        # TODO: Add support for axes arguments later
        min_vals = self.min_vals.zeros_like(*args, **kwargs)
        max_vals = self.max_vals.zeros_like(*args, **kwargs)

        child = (
            np.zeros_like(self.child, *args, **kwargs)
            if isinstance(self.child, np.ndarray)
            else self.child.zeros_like(*args, **kwargs)
        )

        return PhiTensor(
            child=child,
            min_vals=min_vals,
            max_vals=max_vals,
            data_subjects=self.data_subjects,
        )

    def __setitem__(
        self,
        key: Union[int, slice, NDArray],
        value: Union[PhiTensor, GammaTensor, np.ndarray],
    ) -> Union[PhiTensor, GammaTensor]:
        if isinstance(value, PhiTensor):
            self.child[key] = value.child
            minv = value.child.min()
            maxv = value.child.max()

            if minv < self.min_vals.data.min():
                self.min_vals.data = minv

            if maxv > self.max_vals.data.max():
                self.max_vals.data = maxv

            gamma_output = self.gamma
            gamma_output[key] = value.gamma
            # print("It's on the right track")
            return gamma_output

        elif isinstance(value, GammaTensor):
            gamma = self.gamma
            gamma[key] = value
            return gamma
        elif isinstance(value, np.ndarray):
            self.child[key] = value
            minv = value.min()
            maxv = value.max()

            if minv < self.min_vals.data.min():
                self.min_vals.data = minv

            if maxv > self.max_vals.data.max():
                self.max_vals.data = maxv

            return PhiTensor(
                child=self.child,
                data_subjects=self.data_subjects,
                min_vals=self.min_vals,
                max_vals=self.max_vals,
            )
        else:
            raise NotImplementedError

    def __abs__(self) -> PhiTensor:
        data = self.child
        output = np.abs(data)

        min_val = abs(self.min_vals.data)
        max_val = abs(self.max_vals.data)

        new_min_val = min(min_val, max_val)
        new_max_val = max(min_val, max_val)

        return PhiTensor(
            child=output,
            data_subjects=self.data_subjects,
            min_vals=lazyrepeatarray(data=new_min_val, shape=output.shape),
            max_vals=lazyrepeatarray(data=new_max_val, shape=output.shape),
        )

    def argmax(
        self,
        axis: Optional[int] = None,
    ) -> PhiTensor:
        child = self.child.argmax(axis=axis)
        if axis is None:
            max_value = self.child.size - 1
            indices = np.unravel_index(child, shape=self.child.shape)
            data_subjects = self.data_subjects[indices]
        else:
            index = np.array([child])
            max_value = np.size(self.child, axis=axis) - 1
            data_subjects = np.squeeze(
                np.take_along_axis(self.data_subjects, index, axis=axis)
            )

        return PhiTensor(
            child=child,
            data_subjects=data_subjects,
            min_vals=lazyrepeatarray(data=0, shape=child.shape),
            max_vals=lazyrepeatarray(data=max_value, shape=child.shape),
        )

    def argmin(
        self,
        axis: Optional[int] = None,
    ) -> PhiTensor:
        child = self.child.argmin(axis=axis)
        if axis is None:
            max_value = self.child.size - 1
            indices = np.unravel_index(child, shape=self.child.shape)
            data_subjects = self.data_subjects[indices]
        else:
            index = np.array([child])
            max_value = np.size(self.child, axis=axis) - 1
            data_subjects = np.squeeze(
                np.take_along_axis(self.data_subjects, index, axis=axis)
            )

        return PhiTensor(
            child=child,
            data_subjects=data_subjects,
            min_vals=lazyrepeatarray(data=0, shape=child.shape),
            max_vals=lazyrepeatarray(data=max_value, shape=child.shape),
        )

    def reshape(self, *shape: Tuple[int, ...]) -> PhiTensor:

        data = self.child
        output_data = np.reshape(data, *shape)
        return PhiTensor(
            child=output_data,
            data_subjects=np.reshape(self.data_subjects, *shape),
            min_vals=lazyrepeatarray(data=self.min_vals.data, shape=output_data.shape),
            max_vals=lazyrepeatarray(data=self.max_vals.data, shape=output_data.shape),
        )

    def pad(self, width: int, padding_mode: str = "reflect") -> PhiTensor:
        data = self.child

        if padding_mode == "reflect":
            pad_left = pad_right = pad_top = pad_bottom = width
            # RGB image
            if len(data.shape) == 3:
                output_data = np.pad(
                    data,
                    ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                    padding_mode,
                )
                output_data_subjects = np.pad(
                    self.data_subjects,
                    ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                    padding_mode,
                )
            # Grayscale image
            elif len(data.shape) == 2:
                output_data = np.pad(
                    data, ((pad_top, pad_bottom), (pad_left, pad_right)), padding_mode
                )
                output_data_subjects = np.pad(
                    self.data_subjects,
                    ((pad_top, pad_bottom), (pad_left, pad_right)),
                    padding_mode,
                )
            else:
                output_data = np.pad(data, width, padding_mode)
                output_data_subjects = np.pad(self.data_subjects, width, padding_mode)
        else:
            raise NotImplementedError

        output_min_val, output_max_val = output_data.min(), output_data.max()
        return PhiTensor(
            child=output_data,
            data_subjects=output_data_subjects,
            min_vals=output_min_val,
            max_vals=output_max_val,
        )

    def ravel(self, order: Optional[str] = "C") -> PhiTensor:
        data = self.child
        output_data = data.ravel(order=order)

        output_data_subjects = self.data_subjects.ravel(order=order)

        min_vals = lazyrepeatarray(data=self.min_vals.data, shape=output_data.shape)
        max_vals = lazyrepeatarray(data=self.max_vals.data, shape=output_data.shape)

        return PhiTensor(
            child=output_data,
            data_subjects=output_data_subjects,
            min_vals=min_vals,
            max_vals=max_vals,
        )

    def random_horizontal_flip(self, p: float = 0.5) -> PhiTensor:
        """Could make more efficient by not encoding/decoding FPT"""
        if np.random.random() <= p:
            return PhiTensor(
                child=np.fliplr(self.child),
                data_subjects=self.data_subjects,
                min_vals=self.min_vals.horizontal_flip(),
                max_vals=self.max_vals.horizontal_flip(),
            )
        else:
            return self

    def random_vertical_flip(self, p: float = 0.5) -> PhiTensor:
        """Could make more efficient by not encoding/decoding FPT"""
        if np.random.random() <= p:
            return PhiTensor(
                child=np.flipud(self.child),
                data_subjects=self.data_subjects,
                min_vals=self.min_vals.vertical_flip(),
                max_vals=self.max_vals.vertical_flip(),
            )
        else:
            return self

    def random_rotation(self, degrees: Union[int, Tuple]) -> PhiTensor:
        if isinstance(degrees, int):
            angle = np.random.randint(low=-degrees, high=degrees)
        elif isinstance(degrees, tuple):
            angle = np.random.randint(low=degrees[0], high=degrees[1])

        rotated_data_value = rotate(self.child, angle)

        return PhiTensor(
            child=rotated_data_value,
            data_subjects=self.data_subjects,
            min_vals=rotated_data_value.min(),
            max_vals=rotated_data_value.max(),
        )

    def max(self, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> PhiTensor:
        """
        Return the maximum of an array or along an axis.

        Parameters
            axis: None or int or tuple of ints, optional
                Axis or axes along which to operate. By default, flattened input is used.
                If this is a tuple of ints, the minimum is selected over multiple axes,
                instead of a single axis or all the axes as before.

            keepdims: bool, optional
                If this is set to True, the axes which are reduced are left in the result as dimensions with size one.
                With this option, the result will broadcast correctly against the input array.
                If the default value is passed, then keepdims will not be passed through to the amax method of
                sub-classes of ndarray, however any non-default value will be.
                If the sub-class’ method does not implement keepdims any exceptions will be raised.
            initial: scalar, optional
                The minimum value of an output element. Must be present to allow computation on empty slice.
                See reduce for details.

            where: array_like of bool, optional
                Elements to compare for the maximum. See reduce for details.

        Returns
            a_max: PhiTensor
                Maximum of a.
                If axis is None, the result is a scalar value.
                If axis is given, the result is an array of dimension a.ndim - 1.
        """
        indices = np.unravel_index(self.child.argmax(axis), shape=self.child.shape)
        result = self.child.max(axis)
        return PhiTensor(
            child=result,
            data_subjects=self.data_subjects[indices],
            min_vals=lazyrepeatarray(data=self.min_vals.data, shape=result.shape),
            max_vals=lazyrepeatarray(data=self.max_vals.data, shape=result.shape),
        )

    def min(self, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> PhiTensor:
        """
        Return the minimum of an array or minimum along an axis.

        Parameters
            axis: None or int or tuple of ints, optional
                Axis or axes along which to operate. By default, flattened input is used.
                If this is a tuple of ints, the minimum is selected over multiple axes,
                instead of a single axis or all the axes as before.

        Returns
            a_min: PhiTensor
                Minimum of a.
                If axis is None, the result is a scalar value.
                If axis is given, the result is an array of dimension a.ndim - 1.
        """

        indices = np.unravel_index(self.child.argmin(axis), self.child.shape)
        result = self.child.min(axis)
        return PhiTensor(
            child=result,
            data_subjects=self.data_subjects[indices],
            min_vals=lazyrepeatarray(data=self.min_vals.data, shape=result.shape),
            max_vals=lazyrepeatarray(data=self.max_vals.data, shape=result.shape),
        )

    def _argmax(self, axis: Optional[int]) -> PhiTensor:
        return self.child.argmax(axis)

    def unravel_argmax(
        self, axis: Optional[int] = None
    ) -> Tuple[np.ndarray]:  # possible privacy violation?
        arg_result = self._argmax(axis=axis)
        shape = self.shape
        return np.unravel_index(arg_result, shape)

    def swapaxes(self, axis1: int, axis2: int) -> PhiTensor:
        """Interchange two axes of an array."""
        out_child = np.swapaxes(self.child, axis1, axis2)
        data_subjects = np.swapaxes(self.data_subjects, axis1, axis2)
        return PhiTensor(
            child=out_child,
            data_subjects=data_subjects,
            min_vals=lazyrepeatarray(data=self.min_vals.data, shape=out_child.shape),
            max_vals=lazyrepeatarray(data=self.max_vals.data, shape=out_child.shape),
        )

    def nonzero(self) -> PhiTensor:
        """Return the indices of the elements that are non-zero."""
        out_child = np.array(np.nonzero(self.child))
        no_axis = len(self.child.shape)
        out_data_subjects = np.repeat(
            np.array([self.data_subjects[self.child != 0]]), no_axis, axis=0
        )
        return PhiTensor(
            child=out_child,
            data_subjects=out_data_subjects,
            min_vals=lazyrepeatarray(data=0, shape=out_child.shape),
            max_vals=lazyrepeatarray(data=max(self.child.shape), shape=out_child.shape),
        )

    def mean(
        self,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        **kwargs: Any,
    ) -> PhiTensor:
        result = self.child.mean(axis, **kwargs)

        return PhiTensor(
            child=result,
            data_subjects=self.data_subjects.mean(axis, **kwargs),
            min_vals=lazyrepeatarray(data=self.min_vals.data, shape=result.shape),
            max_vals=lazyrepeatarray(data=self.max_vals.data, shape=result.shape),
        )

    def std(
        self,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        **kwargs: Any,
    ) -> PhiTensor:
        """
        Compute the standard deviation along the specified axis.
        Returns the standard deviation, a measure of the spread of a distribution, of the array elements.
        The standard deviation is computed for the flattened array by default, otherwise over the specified axis.

        Parameters
            axis: None or int or tuple of ints, optional
                Axis or axes along which the standard deviation is computed.
                The default is to compute the standard deviation of the flattened array.
                If this is a tuple of ints, a standard deviation is performed over multiple axes, instead of a single
                axis or all the axes as before.

            out: ndarray, optional
                Alternative output array in which to place the result. It must have the same shape as the expected
                output but the type (of the calculated values) will be cast if necessary.

            ddof: int, optional
                ddof = Delta Degrees of Freedom. By default ddof is zero.
                The divisor used in calculations is N - ddof, where N represents the number of elements.

            keepdims: bool, optional
                If this is set to True, the axes which are reduced are left in the result as dimensions with size one.
                With this option, the result will broadcast correctly against the input array.
                If the default value is passed, then keepdims will not be passed through to the std method of
                sub-classes of ndarray, however any non-default value will be. If the sub-class’ method does not
                implement keepdims any exceptions will be raised.

            where: array_like of bool, optional
                Elements to include in the standard deviation. See reduce for details.

        Returns

            standard_deviation: PhiTensor
        """

        result = self.child.std(axis, **kwargs)
        # Std is lowest when all values are the same, 0. (-ve not possible because of squaring)
        # Std is highest when half the samples are min and other half are max
        return PhiTensor(
            child=result,
            data_subjects=self.data_subjects.std(axis, **kwargs),
            min_vals=lazyrepeatarray(data=np.array([0]), shape=result.shape),
            max_vals=lazyrepeatarray(
                data=(self.max_vals.data - self.min_vals.data) / 2, shape=result.shape
            ),
        )

    def var(
        self,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        **kwargs: Any,
    ) -> PhiTensor:
        """
        Compute the variance along the specified axis of the array elements, a measure of the spread of a distribution.
        The variance is computed for the flattened array by default, otherwise over the specified axis.

        Parameters

            axis: None or int or tuple of ints, optional
                Axis or axes along which the variance is computed.
                The default is to compute the variance of the flattened array.
                If this is a tuple of ints, a variance is performed over multiple axes, instead of a single axis or all
                the axes as before.

            ddof: int, optional
                “Delta Degrees of Freedom”: the divisor used in the calculation is N - ddof, where N represents the
                number of elements. By default ddof is zero.

            keepdims: bool, optional
                If this is set to True, the axes which are reduced are left in the result as dimensions with size one.
                With this option, the result will broadcast correctly against the input array.
                If the default value is passed, then keepdims will not be passed through to the var method of
                sub-classes of ndarray, however any non-default value will be. If the sub-class’ method does not
                implement keepdims any exceptions will be raised.

            where: array_like of bool, optional
                Elements to include in the variance. See reduce for details.
        """

        result = self.child.var(axis, **kwargs)
        return PhiTensor(
            child=result,
            data_subjects=self.data_subjects.var(axis, **kwargs),
            min_vals=lazyrepeatarray(data=0, shape=result.shape),
            max_vals=lazyrepeatarray(
                data=0.25 * (self.max_vals.data - self.min_vals.data) ** 2,
                shape=result.shape,
            ),
        )

    def sqrt(self) -> PhiTensor:
        result = np.sqrt(self.child)
        return PhiTensor(
            child=result,
            data_subjects=np.sqrt(self.data_subjects),
            min_vals=lazyrepeatarray(
                data=np.sqrt(self.min_vals.data), shape=result.shape
            ),
            max_vals=lazyrepeatarray(
                data=np.sqrt(self.max_vals.data), shape=result.shape
            ),
        )

    def normalize(
        self, mean: Union[float, Sequence[float]], std: Union[float, Sequence[float]]
    ) -> PhiTensor:
        # TODO: Double check if normalization bounds are correct; they might be data dependent
        if isinstance(mean, float) and isinstance(std, float):
            return PhiTensor(
                child=(self.child - mean) / std,
                data_subjects=self.data_subjects,
                min_vals=(self.min_vals - mean) * (1 / std),
                max_vals=(self.max_vals - mean) * (1 / std),
            )
        else:
            # This is easily doable in the future
            raise NotImplementedError

    def create_gamma(self) -> GammaTensor:
        """Return a new Gamma tensor based on this phi tensor"""
        gamma_tensor = GammaTensor(
            child=self.child,
            data_subjects=self.data_subjects,
            min_vals=self.min_vals,
            max_vals=self.max_vals,
        )

        return gamma_tensor

    def view(self, *args: Any) -> PhiTensor:
        # TODO: Figure out how to fix lazyrepeatarray reshape

        data = self.child.reshape(*args)
        return PhiTensor(
            child=data,
            data_subjects=self.data_subjects,
            min_vals=lazyrepeatarray(data=self.min_vals.data.min(), shape=data.shape),
            max_vals=lazyrepeatarray(data=self.max_vals.data.max(), shape=data.shape),
        )

    def publish(
        self,
        get_budget_for_user: Callable,
        deduct_epsilon_for_user: Callable,
        ledger: DataSubjectLedger,
        sigma: float,
        private: bool,
    ) -> AcceptableSimpleType:
        print("PUBLISHING TO GAMMA:")
        print(self.child)

        gamma = self.gamma
        # gamma.func = lambda x: x
        gamma.sources[gamma.id] = gamma

        res = gamma.publish(
            get_budget_for_user=get_budget_for_user,
            deduct_epsilon_for_user=deduct_epsilon_for_user,
            ledger=ledger,
            sigma=sigma,
            private=private,
        )

        print("Final Values", res)

        return res

    @property
    def value(self) -> np.ndarray:
        return self.child

    def astype(self, np_type: np.dtype) -> PhiTensor:
        return self.__class__(
            child=self.child.astype(np_type),
            data_subjects=self.data_subjects,
            min_vals=self.min_vals.astype(np_type),
            max_vals=self.max_vals.astype(np_type),
            # scalar_manager=self.scalar_manager,
        )

    @property
    def shape(self) -> Tuple[Any, ...]:
        if self.child is None:
            return ()
        else:
            return self.child.shape

    def __repr__(self) -> str:
        """Pretty print some information, optimized for Jupyter notebook viewing."""
        return (
            f"{self.__class__.__name__}(child={self.child}, "
            + f"min_vals={self.min_vals}, max_vals={self.max_vals})"
        )

    def __eq__(self, other: Any) -> Union[PhiTensor, GammaTensor]:  # type: ignore
        if is_acceptable_simple_type(other):
            return PhiTensor(
                child=(self.child == other)
                * 1,  # Multiply by 1 to convert to 0/1 instead of T/F
                data_subjects=self.data_subjects,
                min_vals=lazyrepeatarray(data=0, shape=self.shape),
                max_vals=lazyrepeatarray(data=1, shape=self.shape),
            )
        elif isinstance(other, PhiTensor):
            if (self.data_subjects != other.data_subjects).any():
                return self.gamma == other.gamma
            else:
                return PhiTensor(
                    child=(self.child == other.child)
                    * 1,  # Multiply by 1 to convert to 0/1 instead of T/F
                    data_subjects=self.data_subjects,
                    min_vals=lazyrepeatarray(data=0, shape=self.shape),
                    max_vals=lazyrepeatarray(data=1, shape=self.shape),
                )
        else:
            raise NotImplementedError(
                f"__eq__ not implemented between PhiTensor and {type(other)}."
            )

    def __ne__(self, other: Any) -> Union[PhiTensor, GammaTensor]:  # type: ignore
        if is_acceptable_simple_type(other):
            return PhiTensor(
                child=(self.child != other)
                * 1,  # Multiply by 1 to convert to 0/1 instead of T/F
                data_subjects=self.data_subjects,
                min_vals=lazyrepeatarray(data=0, shape=self.shape),
                max_vals=lazyrepeatarray(data=1, shape=self.shape),
            )
        elif isinstance(other, PhiTensor):
            if (self.data_subjects != other.data_subjects).any():
                return self.gamma != other.gamma
            else:
                return PhiTensor(
                    child=(self.child != other.child)
                    * 1,  # Multiply by 1 to convert to 0/1 instead of T/F
                    data_subjects=self.data_subjects,
                    min_vals=lazyrepeatarray(data=0, shape=self.shape),
                    max_vals=lazyrepeatarray(data=1, shape=self.shape),
                )
        else:
            raise NotImplementedError(
                f"__ne__ not implemented between PhiTensor and {type(other)}."
            )

    def __add__(self, other: SupportedChainType) -> Union[PhiTensor, GammaTensor]:

        # if the tensor being added is also private
        if isinstance(other, PhiTensor):
            return self.gamma + other.gamma
            # if self.data_subjects != other.data_subjects:
            #     return self.gamma + other.gamma

            # return PhiTensor(
            #     child=self.child + other.child,
            #     min_vals=self.min_vals + other.min_vals,
            #     max_vals=self.max_vals + other.max_vals,
            #     data_subjects=self.data_subjects,
            #     # scalar_manager=self.scalar_manager,
            # )

        # if the tensor being added is a public tensor / int / float / etc.
        elif is_acceptable_simple_type(other):

            return PhiTensor(
                child=self.child + other,
                min_vals=self.min_vals + other,
                max_vals=self.max_vals + other,
                data_subjects=self.data_subjects,
            )

        elif isinstance(other, GammaTensor):
            return self.gamma + other
        else:
            print("Type is unsupported:" + str(type(other)))
            raise NotImplementedError

    def __sub__(self, other: SupportedChainType) -> Union[PhiTensor, GammaTensor]:

        if isinstance(other, PhiTensor):
            return self.gamma - other.gamma
            # diff_data_subjects = (
            #     self.data_subjects.one_hot_lookup != other.data_subjects.one_hot_lookup
            # )
            # diff_data_subjects = (
            #     diff_data_subjects
            #     if isinstance(diff_data_subjects, bool)
            #     else diff_data_subjects.any()
            # )
            # if diff_data_subjects:
            #     return self.gamma - other.gamma
            #     # raise NotImplementedError

            # data = self.child - other.child
            # min_min = self.min_vals.data - other.min_vals.data
            # min_max = self.min_vals.data - other.max_vals.data
            # max_min = self.max_vals.data - other.min_vals.data
            # max_max = self.max_vals.data - other.max_vals.data
            # _min_vals = np.minimum.reduce([min_min, min_max, max_min, max_max])
            # _max_vals = np.maximum.reduce([min_min, min_max, max_min, max_max])
            # min_vals = self.min_vals.copy()
            # min_vals.data = _min_vals
            # max_vals = self.max_vals.copy()
            # max_vals.data = _max_vals

            # data_subjects = self.data_subjects

        elif is_acceptable_simple_type(other):
            if isinstance(other, np.ndarray):
                if not is_broadcastable(other.shape, self.child.shape):  # type: ignore
                    raise Exception(
                        f"Shapes do not match for subtraction: {self.child.shape} and {other.shape}"
                    )
            data = self.child - other
            min_vals = self.min_vals - other
            max_vals = self.max_vals - other
            data_subjects = self.data_subjects
        elif isinstance(other, GammaTensor):
            return self.gamma - other
        else:
            print("Type is unsupported:" + str(type(other)))
            raise NotImplementedError
        return PhiTensor(
            child=data,
            data_subjects=data_subjects,
            min_vals=min_vals,
            max_vals=max_vals,
        )

    def __mul__(self, other: SupportedChainType) -> Union[PhiTensor, GammaTensor]:

        if isinstance(other, PhiTensor):
            if np.array(self.data_subjects == other.data_subjects).all():
                min_min = self.min_vals.data * other.min_vals.data
                min_max = self.min_vals.data * other.max_vals.data
                max_min = self.max_vals.data * other.min_vals.data
                max_max = self.max_vals.data * other.max_vals.data

                _min_vals = np.min([min_min, min_max, max_min, max_max], axis=0)  # type: ignore
                _max_vals = np.max([min_min, min_max, max_min, max_max], axis=0)  # type: ignore

                return PhiTensor(
                    child=self.child * other.child,
                    data_subjects=self.data_subjects,
                    min_vals=lazyrepeatarray(data=_min_vals, shape=self.shape),
                    max_vals=lazyrepeatarray(data=_max_vals, shape=self.shape),
                )
            else:
                return self.gamma * other.gamma

        elif is_acceptable_simple_type(other):

            data = self.child * other

            min_min = self.min_vals.data * other
            min_max = self.min_vals.data * other
            max_min = self.max_vals.data * other
            max_max = self.max_vals.data * other

            _min_vals = np.min([min_min, min_max, max_min, max_max], axis=0)  # type: ignore
            _max_vals = np.max([min_min, min_max, max_min, max_max], axis=0)  # type: ignore
            min_vals = self.min_vals.copy()
            min_vals.data = _min_vals
            max_vals = self.max_vals.copy()
            max_vals.data = _max_vals

            data_subjects = self.data_subjects

            return PhiTensor(
                child=data,
                data_subjects=data_subjects,
                min_vals=min_vals,
                max_vals=max_vals,
            )
        elif isinstance(other, GammaTensor):
            return self.gamma * other
        else:
            print("Type is unsupported:" + str(type(other)))
            raise NotImplementedError

    def __truediv__(self, other: Any) -> Union[PhiTensor, GammaTensor]:
        if isinstance(other, PhiTensor):
            if np.array(self.data_subjects != other.data_subjects).all():
                return self.gamma / other.gamma
            else:
                min_min = self.min_vals.data / other.min_vals.data
                min_max = self.min_vals.data / other.max_vals.data
                max_min = self.max_vals.data / other.min_vals.data
                max_max = self.max_vals.data / other.max_vals.data

                _min_vals = np.min([min_min, min_max, max_min, max_max], axis=0)  # type: ignore
                _max_vals = np.max([min_min, min_max, max_min, max_max], axis=0)  # type: ignore

                return PhiTensor(
                    child=self.child / other.child,
                    data_subjects=self.data_subjects,
                    min_vals=lazyrepeatarray(data=_min_vals, shape=self.shape),
                    max_vals=lazyrepeatarray(data=_max_vals, shape=self.shape),
                )
        elif isinstance(other, GammaTensor):
            return self.gamma / other
        elif is_acceptable_simple_type(other):
            return PhiTensor(
                child=self.child / other,
                data_subjects=self.data_subjects,
                min_vals=lazyrepeatarray(
                    data=self.min_vals.data / other, shape=self.min_vals.shape
                ),
                max_vals=lazyrepeatarray(
                    data=self.max_vals.data / other, shape=self.max_vals.shape
                ),
            )
        else:
            raise NotImplementedError(
                f"truediv not supported between PhiTensor & {type(other)}"
            )

    def __rtruediv__(self, other: SupportedChainType) -> Union[PhiTensor, GammaTensor]:

        if is_acceptable_simple_type(other):
            return PhiTensor(
                child=(1 / self.child) * other,
                min_vals=(1 / self.min_vals) * other,
                max_vals=(1 / self.max_vals) * other,
                data_subjects=self.data_subjects,
            )

        elif isinstance(other, GammaTensor):
            return (1 / self.gamma) * other
        else:
            print("Type is unsupported:" + str(type(other)))
            raise NotImplementedError

    def __matmul__(
        self, other: Union[np.ndarray, PhiTensor]
    ) -> Union[PhiTensor, GammaTensor]:
        if not isinstance(other, (np.ndarray, PhiTensor, GammaTensor)):
            raise Exception(
                f"Matrix multiplication not yet implemented for type {type(other)}"
            )
        else:
            if isinstance(other, np.ndarray):
                data = self.child.__matmul__(other)
                min_min = (self.min_vals @ other).data
                min_max = (self.min_vals @ other).data
                max_max = (self.max_vals @ other).data
                max_min = (self.max_vals @ other).data
                minv = np.min([min_min, min_max, max_max, max_min], axis=0)  # type: ignore
                min_vals = lazyrepeatarray(data=minv, shape=data.shape)
                max_vals = self.max_vals.__matmul__(other)
                output_ds = self.data_subjects @ other
            elif isinstance(other, PhiTensor):
                if self.data_subjects.sum() != other.data_subjects.sum():
                    return self.gamma @ other.gamma
                else:
                    min_min = (self.min_vals @ other.min_vals).data.min()
                    min_max = (self.min_vals @ other.max_vals).data.min()
                    max_max = (self.max_vals @ other.max_vals).data.min()
                    max_min = (self.max_vals @ other.min_vals).data.min()

                    if self.min_vals.data <= 0 and self.max_vals.data >= 0:
                        minv = np.min([0, min_min, min_max, max_max, max_min], axis=0)  # type: ignore
                    else:
                        minv = np.min([min_min, min_max, max_max, max_min], axis=0)  # type: ignore

                    data = self.child.__matmul__(other.child)
                    max_vals = self.max_vals.__matmul__(other.max_vals)
                    min_vals = lazyrepeatarray(data=minv, shape=data.shape)
                    output_ds = self.data_subjects @ other.data_subjects

            elif isinstance(other, GammaTensor):
                return self.gamma @ other
            else:
                print("Type is unsupported:" + str(type(other)))
                raise NotImplementedError

            return PhiTensor(
                child=data,
                max_vals=max_vals,
                min_vals=min_vals,
                data_subjects=output_ds,
            )

    def argsort(self, axis: Optional[int] = -1) -> PhiTensor:
        """
        Returns the indices that would sort an array.

        Perform an indirect sort along the given axis using the algorithm specified by the kind keyword.
        It returns an array of indices of the same shape as a that index data along the given axis in sorted order.

        Parameters
            axis: int or None, optional
                Axis along which to sort. The default is -1 (the last axis). If None, the flattened array is used.
            kind: {‘quicksort’, ‘mergesort’, ‘heapsort’, ‘stable’}, optional
                Sorting algorithm. The default is ‘quicksort’. Note that both ‘stable’ and ‘mergesort’ use timsort
                under the covers and, in general, the actual implementation will vary with data type. The ‘mergesort’
                option is retained for backwards compatibility.
            order: str or list of str, optional
                When a is an array with fields defined, this argument specifies which fields to compare 1st, 2nd, etc.
                A single field can be specified as a string, and not all fields need be specified, but unspecified
                fields will still be used, in the order in which they come up in the dtype, to break ties.

        Returns
            index_array: ndarray, int
                Array of indices that sort a along the specified axis. If a is one-dimensional, a[index_array] yields a
                sorted a. More generally, np.take_along_axis(a, index_array, axis=axis) always yields the sorted a,
                irrespective of dimensionality.
        """
        result = self.child.argsort(axis)
        out_ds = np.take_along_axis(self.data_subjects, result, axis=axis)
        return PhiTensor(
            child=result,
            data_subjects=out_ds,
            min_vals=lazyrepeatarray(data=0, shape=self.shape),
            max_vals=lazyrepeatarray(data=self.child.size, shape=self.shape),
        )

    def sort(self, axis: int = -1, kind: Optional[str] = None) -> PhiTensor:
        """
        Return a sorted copy of an array.

        Parameters

            a: array_like
                Array to be sorted.

            axis: int or None, optional
                Axis along which to sort. If None, the array is flattened before sorting.
                The default is -1, which sorts along the last axis.

            kind{‘quicksort’, ‘mergesort’, ‘heapsort’, ‘stable’}, optional
                Sorting algorithm. The default is ‘quicksort’.
                Note that both ‘stable’ and ‘mergesort’ use timsort or radix sort under the covers and, in general,
                the actual implementation will vary with data type. The ‘mergesort’ option is retained for backwards
                compatibility.

                Changed in version 1.15.0.: The ‘stable’ option was added.

            order: str or list of str, optional
                When a is an array with fields defined, this argument specifies which fields to compare first, second,
                etc. A single field can be specified as a string, and not all fields need be specified, but unspecified
                 fields will still be used, in the order in which they come up in the dtype, to break ties.

        Please see docs here: https://numpy.org/doc/stable/reference/generated/numpy.sort.html
        """

        # Must do argsort before we change self.child by calling sort
        indices = self.child.argsort(axis, kind)
        self.child.sort(axis, kind)

        out_ds = np.take_along_axis(self.data_subjects, indices, axis=axis)
        return PhiTensor(
            child=self.child,
            data_subjects=out_ds,
            min_vals=self.min_vals,
            max_vals=self.max_vals,
        )

    def __lshift__(self, other: Any) -> Union[PhiTensor, GammaTensor]:
        if is_acceptable_simple_type(other):
            if isinstance(other, np.ndarray):
                other_max, other_min = other.max(), other.min()
            else:
                other_max, other_min = other, other

            child = self.child << other

        elif isinstance(other, GammaTensor):
            return self.gamma << other
        elif isinstance(other, PhiTensor):
            if (self.data_subjects != other.data_subjects).any():
                return self.gamma << other.gamma
            else:
                child = self.child << other.child
                other_max = other.max_vals.data
                other_min = other.min_vals.data
        else:
            raise NotImplementedError(
                f"__lshift__ not implemented between PhiTensor and {type(other)}."
            )

        min_min = self.min_vals.data << other_min
        min_max = self.min_vals.data << other_max
        max_min = self.max_vals.data << other_min
        max_max = self.max_vals.data << other_max

        _min_vals = np.min([min_min, min_max, max_min, max_max], axis=0)  # type: ignore
        _max_vals = np.max([min_min, min_max, max_min, max_max], axis=0)  # type: ignore
        return PhiTensor(
            child=child,
            data_subjects=self.data_subjects,
            min_vals=lazyrepeatarray(data=_min_vals, shape=self.shape),
            max_vals=lazyrepeatarray(data=_max_vals, shape=self.shape),
        )

    def __rshift__(self, other: Any) -> Union[PhiTensor, GammaTensor]:
        if is_acceptable_simple_type(other):
            if isinstance(other, np.ndarray):
                other_max, other_min = other.max(), other.min()
            else:
                other_max, other_min = other, other
            child = self.child >> other

        elif isinstance(other, GammaTensor):
            return self.gamma >> other
        elif isinstance(other, PhiTensor):
            if (self.data_subjects != other.data_subjects).any():
                return self.gamma >> other.gamma
            else:
                child = self.child >> other.child
                other_max = other.max_vals.data
                other_min = other.min_vals.data
        else:
            raise NotImplementedError(
                f"__rshift__ not implemented between PhiTensor and {type(other)}."
            )

        min_min = self.min_vals.data >> other_min
        min_max = self.min_vals.data >> other_max
        max_min = self.max_vals.data >> other_min
        max_max = self.max_vals.data >> other_max

        _min_vals = np.min([min_min, min_max, max_min, max_max], axis=0)  # type: ignore
        _max_vals = np.max([min_min, min_max, max_min, max_max], axis=0)  # type: ignore
        return PhiTensor(
            child=child,
            data_subjects=self.data_subjects,
            min_vals=lazyrepeatarray(data=_min_vals, shape=self.shape),
            max_vals=lazyrepeatarray(data=_max_vals, shape=self.shape),
        )

    def __xor__(self, other: Any) -> Union[PhiTensor, GammaTensor]:
        if is_acceptable_simple_type(other):
            if isinstance(other, np.ndarray):
                other_min, other_max = other.min(), other.max()
            else:
                other_min, other_max = other, other
            child = self.child ^ other
        elif isinstance(other, GammaTensor):
            return self.gamma ^ other
        elif isinstance(other, PhiTensor):
            if (self.data_subjects != other.data_subjects).any():
                return self.gamma ^ other.gamma
            else:
                child = self.child ^ other.child
                other_min, other_max = other.min_vals.data, other.max_vals.data
        else:
            raise NotImplementedError(
                f"__xor__ not implemented between PhiTensor and {type(other)}."
            )

        # TODO: should modify for a tighter found for xor
        _max = int(max(self.max_vals.data, other_max))
        _min = int(min(self.min_vals.data, other_min))
        _max_vals = max(
            (2 ** (_min ^ _max).bit_length()) - 1, (2 ** (_max).bit_length()) - 1
        )
        _min_vals = min(0, _min)

        return PhiTensor(
            child=child,
            data_subjects=self.data_subjects,
            min_vals=lazyrepeatarray(data=_min_vals, shape=self.shape),
            max_vals=lazyrepeatarray(data=_max_vals, shape=self.shape),
        )

    def searchsorted(self, v: Any) -> Union[PhiTensor, GammaTensor]:
        """
        https://numpy.org/doc/stable/reference/generated/numpy.searchsorted.html
        """
        raise NotImplementedError

    def __divmod__(
        self, other: Any
    ) -> Tuple[Union[PhiTensor, GammaTensor], Union[PhiTensor, GammaTensor]]:
        if is_acceptable_simple_type(other) or isinstance(
            other, (PhiTensor, GammaTensor)
        ):
            return self // other, self % other  # type: ignore
        else:
            raise NotImplementedError(
                f"PhiTensor divmod not supported for type: {other}"
            )

    def divmod(
        self, other: Any
    ) -> Tuple[Union[PhiTensor, GammaTensor], Union[PhiTensor, GammaTensor]]:
        return self.__divmod__(other)

    def __round__(self, n: int = 0) -> PhiTensor:
        return PhiTensor(
            child=self.child.round(n),
            data_subjects=self.data_subjects,
            min_vals=lazyrepeatarray(
                data=self.min_vals.data.round(n), shape=self.min_vals.shape
            ),
            max_vals=lazyrepeatarray(
                data=self.max_vals.data.round(n), shape=self.max_vals.shape
            ),
        )

    def round(self, n: int = 0) -> PhiTensor:
        return self.__round__(n)

    def __rmatmul__(
        self, other: Union[np.ndarray, PhiTensor]
    ) -> Union[PhiTensor, GammaTensor]:
        if not isinstance(other, (np.ndarray, PhiTensor, GammaTensor)):
            raise Exception(
                f"Matrix multiplication not yet implemented for type {type(other)}"
            )
        else:
            # Modify before merge, to know is broadcast is actually necessary
            if False:  # and not is_broadcastable(self.shape, other.shape):
                raise Exception(
                    f"Shapes not broadcastable: {self.shape} and {other.shape}"
                )
            else:
                if isinstance(other, np.ndarray):
                    data = self.child.__rmatmul__(other)
                    min_vals = self.min_vals.__rmatmul__(other)
                    max_vals = self.max_vals.__rmatmul__(other)
                    output_ds = self.data_subjects.__rmatmul__(other)
                elif isinstance(other, PhiTensor):
                    return self.gamma.__rmatmul__(other.gamma)
                    # if self.data_subjects != other.data_subjects:
                    #     # return convert_to_gamma_tensor(self).__matmul__(convert_to_gamma_tensor(other))
                    #     raise NotImplementedError
                    # else:
                    #     data = self.child.__rmatmul__(other.child)
                    #     # _min_vals = np.array(
                    #     #     [self.min_vals.data.__matmul__(other.min_vals.data)]
                    #     # )
                    #     # _max_vals = np.array(
                    #     #     [self.max_vals.data.__matmul__(other.max_vals.data)]
                    #     # )
                    #     # min_vals = self.min_vals.copy()
                    #     # min_vals.data = _min_vals
                    #     # max_vals = self.max_vals.copy()
                    #     # max_vals.data = _max_vals
                    #     min_vals = self.min_vals.__rmatmul__(other.min_vals)
                    #     max_vals = self.max_vals.__rmatmul__(other.max_vals)

                else:
                    print("Type is unsupported:" + str(type(other)))
                    raise NotImplementedError

                return PhiTensor(
                    child=data,
                    max_vals=max_vals,
                    min_vals=min_vals,
                    data_subjects=output_ds,
                )

    def clip(self, a_min: float, a_max: float) -> PhiTensor:
        output_data = np.clip(self.child, a_min, a_max)

        min_v = np.clip(self.min_vals.data, a_min, a_max)
        max_v = np.clip(self.max_vals.data, a_min, a_max)

        min_vals = lazyrepeatarray(data=min_v, shape=output_data.shape)
        max_vals = lazyrepeatarray(data=max_v, shape=output_data.shape)

        return PhiTensor(
            child=output_data,
            data_subjects=self.data_subjects,
            min_vals=min_vals,
            max_vals=max_vals,
        )

    def transpose(self, *args: Any, **kwargs: Any) -> PhiTensor:
        """Transposes self.child, min_vals, and max_vals if these can be transposed, otherwise doesn't change them."""
        output_data = self.child.transpose(*args, **kwargs)

        min_vals = lazyrepeatarray(data=self.min_vals.data, shape=output_data.shape)
        max_vals = lazyrepeatarray(data=self.max_vals.data, shape=output_data.shape)
        output_ds = self.data_subjects.transpose(*args, **kwargs)

        return PhiTensor(
            child=output_data,
            data_subjects=output_ds,
            min_vals=min_vals,
            max_vals=max_vals,
        )

    def flatten(self, order: Optional[str] = "C") -> PhiTensor:
        """
        Return a copy of the array collapsed into one dimension.

        Parameters
            order{‘C’, ‘F’, ‘A’, ‘K’}, optional
                ‘C’ means to flatten in row-major (C-style) order.
                ‘F’ means to flatten in column-major (Fortran- style) order.
                ‘A’ means to flatten in column-major order if a is Fortran contiguous in memory,
                        row-major order otherwise.
                ‘K’ means to flatten a in the order the elements occur in memory. The default is ‘C’.
        Returns
            PhiTensor
                A copy of the input array, flattened to one dimension.
        """
        if order not in ["C", "F", "A", "K"]:
            raise NotImplementedError(f"Flatten is not implemented for order={order}")

        output_data = self.child.flatten(order=order)
        return PhiTensor(
            child=output_data,
            data_subjects=self.data_subjects.flatten(order=order),
            min_vals=lazyrepeatarray(data=self.min_vals.data, shape=output_data.shape),
            max_vals=lazyrepeatarray(data=self.max_vals.data, shape=output_data.shape),
        )

    def concatenate(
        self,
        other: Union[np.ndarray, PhiTensor],
        *args: Any,
        **kwargs: Any,
    ) -> Union[PhiTensor, GammaTensor]:

        # if the tensor being added is also private
        if isinstance(other, PhiTensor):
            if self.data_subjects != other.data_subjects:
                return self.gamma + other.gamma

            return PhiTensor(
                child=self.child.concatenate(other.child, *args, **kwargs),
                min_vals=self.min_vals.concatenate(other.min_vals, *args, **kwargs),
                max_vals=self.max_vals.concatenate(other.max_vals, *args, **kwargs),
                data_subjects=self.data_subjects,
            )

        elif is_acceptable_simple_type(other):
            raise NotImplementedError
        else:
            print("Type is unsupported:" + str(type(other)))
            raise NotImplementedError

    def __lt__(self, other: SupportedChainType) -> Union[PhiTensor, GammaTensor]:
        if isinstance(other, PhiTensor):
            if np.array(self.data_subjects == other.data_subjects).all():
                return PhiTensor(
                    child=(self.child < other.child) * 1,
                    data_subjects=self.data_subjects,
                    min_vals=lazyrepeatarray(data=0, shape=self.shape),
                    max_vals=lazyrepeatarray(data=1, shape=self.shape),
                )
            else:
                return self.gamma.__lt__(other.gamma)
        elif isinstance(other, GammaTensor):
            return self.gamma.__lt__(other)

        # if the tensor being compared is a public tensor / int / float / etc.
        elif is_acceptable_simple_type(other):

            data = self.child < other
            min_vals = self.min_vals * 0
            max_vals = (self.max_vals * 0) + 1
            data_subjects = self.data_subjects

            return PhiTensor(
                child=data,
                data_subjects=data_subjects,
                min_vals=min_vals,
                max_vals=max_vals,
            )

        else:
            raise NotImplementedError

    def __le__(self, other: SupportedChainType) -> Union[PhiTensor, GammaTensor]:

        # if the tensor being compared is also private
        if isinstance(other, PhiTensor):
            if np.array(self.data_subjects == other.data_subjects).all():
                return PhiTensor(
                    child=(self.child <= other.child) * 1,
                    data_subjects=self.data_subjects,
                    min_vals=lazyrepeatarray(data=0, shape=self.shape),
                    max_vals=lazyrepeatarray(data=1, shape=self.shape),
                )
            else:
                return self.gamma.__le__(other.gamma)
        elif isinstance(other, GammaTensor):
            return self.gamma.__le__(other)

        # if the tensor being compared is a public tensor / int / float / etc.
        elif is_acceptable_simple_type(other):

            data = self.child <= other
            min_vals = self.min_vals * 0
            max_vals = (self.max_vals * 0) + 1
            data_subjects = self.data_subjects

            return PhiTensor(
                child=data,
                data_subjects=data_subjects,
                min_vals=min_vals,
                max_vals=max_vals,
            )

        else:
            raise NotImplementedError

    def __gt__(self, other: SupportedChainType) -> Union[PhiTensor, GammaTensor]:

        # if the tensor being compared is also private
        if isinstance(other, PhiTensor):
            if np.array(self.data_subjects == other.data_subjects).all():
                return PhiTensor(
                    child=(self.child > other.child) * 1,
                    data_subjects=self.data_subjects,
                    min_vals=lazyrepeatarray(data=0, shape=self.shape),
                    max_vals=lazyrepeatarray(data=1, shape=self.shape),
                )
            else:
                return self.gamma.__gt__(other.gamma)
        elif isinstance(other, GammaTensor):
            return self.gamma.__gt__(other)

        # if the tensor being compared is a public tensor / int / float / etc.
        elif is_acceptable_simple_type(other):

            data = self.child > other
            min_vals = self.min_vals * 0
            max_vals = (self.max_vals * 0) + 1
            data_subjects = self.data_subjects

            return PhiTensor(
                child=data,
                data_subjects=data_subjects,
                min_vals=min_vals,
                max_vals=max_vals,
            )
        else:
            raise NotImplementedError  # type: ignore

    def __ge__(self, other: SupportedChainType) -> Union[PhiTensor, GammaTensor]:

        # if the tensor being compared is also private
        if isinstance(other, PhiTensor):
            if np.array(self.data_subjects == other.data_subjects).all():
                return PhiTensor(
                    child=(self.child >= other.child) * 1,
                    data_subjects=self.data_subjects,
                    min_vals=lazyrepeatarray(data=0, shape=self.shape),
                    max_vals=lazyrepeatarray(data=1, shape=self.shape),
                )
            else:
                return self.gamma.__ge__(other.gamma)
        elif isinstance(other, GammaTensor):
            return self.gamma.__ge__(other)

        # if the tensor being compared is a public tensor / int / float / etc.
        elif is_acceptable_simple_type(other):

            data = self.child >= other
            min_vals = self.min_vals * 0
            max_vals = (self.max_vals * 0) + 1
            data_subjects = self.data_subjects

            return PhiTensor(
                child=data,
                data_subjects=data_subjects,
                min_vals=min_vals,
                max_vals=max_vals,
            )

        else:
            return NotImplementedError  # type: ignore

    # Re enable after testing
    def dot(
        self, other: Union[PhiTensor, GammaTensor, np.ndarray]
    ) -> Union[PhiTensor, GammaTensor]:
        if isinstance(other, np.ndarray):
            return PhiTensor(
                child=np.dot(self.child, other),
                min_vals=np.dot(self.min_vals, other),
                max_vals=np.dot(self.max_vals, other),
                data_subjects=np.dot(self.data_subjects, other),
            )
        elif isinstance(other, PhiTensor):
            return self.gamma.dot(other.gamma)
            # if self.data_subjects.one_hot_lookup == other.data_subjects.one_hot_lookup:
            #     return PhiTensor(
            #         child=np.dot(self.child, other.child),
            #         min_vals=np.dot(self.min_vals, other.min_vals),
            #         max_vals=np.dot(self.max_vals, other.max_vals),
            #         data_subjects=self.data_subjects,
            #     )
            # else:
            #     return self.gamma.dot(other.gamma)
        elif isinstance(other, GammaTensor):
            return self.gamma.dot(other)
        else:
            raise NotImplementedError

    def sum(
        self,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: Optional[bool] = False,
        initial: Optional[float] = None,
        where: Optional[ArrayLike] = None,
    ) -> PhiTensor:
        """
        Sum of array elements over a given axis.

        Parameters
            axis: None or int or tuple of ints, optional
                Axis or axes along which a sum is performed.
                The default, axis=None, will sum all of the elements of the input array.
                If axis is negative it counts from the last to the first axis.
                If axis is a tuple of ints, a sum is performed on all of the axes specified in the tuple instead of a
                single axis or all the axes as before.
            keepdims: bool, optional
                If this is set to True, the axes which are reduced are left in the result as dimensions with size one.
                With this option, the result will broadcast correctly against the input array.
                If the default value is passed, then keepdims will not be passed through to the sum method of
                sub-classes of ndarray, however any non-default value will be. If the sub-class’ method does not
                implement keepdims any exceptions will be raised.
            initial: scalar, optional
                Starting value for the sum. See reduce for details.
            where: array_like of bool, optional
                Elements to include in the sum. See reduce for details.
        """
        if where is None:
            result = np.array(self.child.sum(axis=axis, keepdims=keepdims))
            output_ds = self.data_subjects.sum(axis=axis, keepdims=keepdims)
            num = np.ones_like(self.child).sum(axis=axis, keepdims=keepdims)
        else:
            result = self.child.sum(axis=axis, keepdims=keepdims, where=where)
            output_ds = self.data_subjects.sum(
                axis=axis, keepdims=keepdims, initial=initial, where=where
            )
            num = np.ones_like(self.child).sum(
                axis=axis, keepdims=keepdims, initial=initial, where=where
            )

        return PhiTensor(
            child=result,
            data_subjects=np.array(output_ds),
            min_vals=lazyrepeatarray(data=self.min_vals.data * num, shape=result.shape),
            max_vals=lazyrepeatarray(data=self.max_vals.data * num, shape=result.shape),
        )

    def __pow__(
        self, power: Union[float, int], modulo: Optional[int] = None
    ) -> PhiTensor:
        if modulo is None:
            if self.min_vals.data <= 0 <= self.max_vals.data:
                # If data is in range [-5, 5], it's possible the minimum is 0 and not (-5)^2
                minv = min(0, (self.min_vals.data**power).min())
            else:
                minv = self.min_vals.data**power

            return PhiTensor(
                child=self.child**power,
                data_subjects=self.data_subjects,
                min_vals=lazyrepeatarray(data=minv, shape=self.shape),
                max_vals=lazyrepeatarray(
                    data=self.max_vals.data**power, shape=self.shape
                ),
            )
        else:
            # This may be unnecessary- modulo is NotImplemented in ndarray.pow
            if self.min_vals.data <= 0 <= self.max_vals.data:
                # If data is in range [-5, 5], it's possible the minimum is 0 and not (-5)^2
                minv = min(0, (self.min_vals.data**power).min() % modulo)
            else:
                minv = (self.min_vals.data**power) % modulo
            return PhiTensor(
                child=self.child**power % modulo,
                data_subjects=self.data_subjects,
                min_vals=lazyrepeatarray(data=minv, shape=self.shape),
                max_vals=lazyrepeatarray(
                    data=(self.max_vals.data**power) % modulo, shape=self.shape
                ),
            )

    def expand_dims(self, axis: int) -> PhiTensor:
        result = np.expand_dims(self.child, axis=axis)
        minv = self.min_vals.copy()
        minv.shape = result.shape
        maxv = self.max_vals.copy()
        maxv.shape = result.shape

        return PhiTensor(
            child=result,
            min_vals=minv,
            max_vals=maxv,
            data_subjects=np.expand_dims(self.data_subjects, axis=axis),
        )

    def ones_like(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Union[PhiTensor, GammaTensor]:
        # TODO: Add support for axes arguments later
        min_vals = self.min_vals.ones_like(*args, **kwargs)
        max_vals = self.max_vals.ones_like(*args, **kwargs)

        child = (
            np.ones_like(self.child, *args, **kwargs)
            if isinstance(self.child, np.ndarray)
            else self.child.ones_like(*args, **kwargs)
        )

        return PhiTensor(
            child=child,
            min_vals=min_vals,
            max_vals=max_vals,
            data_subjects=self.data_subjects,
        )

    def __neg__(self) -> PhiTensor:

        return PhiTensor(
            child=self.child * -1,
            min_vals=self.max_vals * -1,
            max_vals=self.min_vals * -1,
            data_subjects=self.data_subjects,
        )

    def __pos__(self) -> PhiTensor:
        return PhiTensor(
            child=self.child,
            min_vals=self.min_vals,
            max_vals=self.max_vals,
            data_subjects=self.data_subjects,
        )

    def resize(
        self, new_shape: Union[int, Tuple[int, ...]], refcheck: bool = True
    ) -> PhiTensor:
        self.child.resize(new_shape, refcheck=refcheck)
        self.data_subjects.resize(new_shape, refcheck=refcheck)
        self.data_subjects = DataSubjectArray.from_objs(self.data_subjects)
        out_shape = self.child.shape
        return PhiTensor(
            child=self.child,
            min_vals=lazyrepeatarray(data=self.min_vals.data, shape=out_shape),
            max_vals=lazyrepeatarray(data=self.max_vals.data, shape=out_shape),
            data_subjects=self.data_subjects,
        )

    def compress(self, condition: List[bool], axis: Optional[int] = None) -> PhiTensor:
        out_child = self.child.compress(condition, axis)
        if 0 in out_child.shape:
            raise NotImplementedError
        return PhiTensor(
            child=out_child,
            min_vals=lazyrepeatarray(data=self.min_vals.data, shape=out_child.shape),
            max_vals=lazyrepeatarray(data=self.max_vals.data, shape=out_child.shape),
            data_subjects=self.data_subjects.compress(condition, axis),
        )

    def squeeze(self, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> PhiTensor:
        out_child = self.child.squeeze(axis)
        return PhiTensor(
            child=out_child,
            min_vals=lazyrepeatarray(data=self.min_vals.data, shape=out_child.shape),
            max_vals=lazyrepeatarray(data=self.max_vals.data, shape=out_child.shape),
            data_subjects=np.squeeze(self.data_subjects, axis),
        )

    def repeat(
        self, repeats: Union[int, Tuple[int, ...]], axis: Optional[int] = None
    ) -> PhiTensor:
        """
        Repeat elements of an array.

        Parameters
            repeats: int or array of ints

                The number of repetitions for each element. repeats is broadcasted to fit the shape of the given axis.
            axis: int, optional

                The axis along which to repeat values.
                By default, use the flattened input array, and return a flat output array.

        Returns

            repeated_array: PhiTensor

                Output array which has the same shape as a, except along the given axis.

        """

        result = self.child.repeat(repeats, axis)
        if isinstance(self.min_vals, lazyrepeatarray):
            minv = lazyrepeatarray(data=self.min_vals.data.min(), shape=result.shape)
            maxv = lazyrepeatarray(data=self.max_vals.data.max(), shape=result.shape)
        else:
            minv = self.min_vals
            maxv = self.max_vals

        return PhiTensor(
            child=result,
            data_subjects=self.data_subjects.repeat(repeats, axis),
            min_vals=minv,
            max_vals=maxv,
        )

    def choose(
        self,
        choices: Union[Sequence, np.ndarray, PassthroughTensor],
        mode: Optional[str] = "raise",
    ) -> Union[PhiTensor, GammaTensor]:
        """
        Construct an array from an index array and a list of arrays to choose from.

        First of all, if confused or uncertain, definitely look at the Examples - in its full generality,
        this function is less simple than it might seem from the following code description
        (below ndi = numpy.lib.index_tricks):

        np.choose(a,c) == np.array([c[a[I]][I] for I in ndi.ndindex(a.shape)]).

        But this omits some subtleties. Here is a fully general summary:

        Given an “index” array (a) of integers and a sequence of n arrays (choices), a and each choice array are first
        broadcast, as necessary, to arrays of a common shape; calling these Ba and Bchoices[i], i = 0,…,n-1 we have that
         necessarily, Ba.shape == Bchoices[i].shape for each i. Then, a new array with shape Ba.shape is created
         as follows:

            if mode='raise' (the default), then, first of all, each element of a (and thus Ba) must be in the range
            [0, n-1]; now, suppose that i (in that range) is the value at the (j0, j1, ..., jm) position in Ba -
            then the value at the same position in the new array is the value in Bchoices[i] at that same position;

            if mode='wrap', values in a (and thus Ba) may be any (signed) integer; modular arithmetic is used to map
            integers outside the range [0, n-1] back into that range; and then the new array is constructed as above;

            if mode='clip', values in a (and thus Ba) may be any (signed) integer; negative integers are mapped to 0;
            values greater than n-1 are mapped to n-1; and then the new array is constructed as above.

        Parameters

            choices: sequence of arrays

                Choice arrays. a and all of the choices must be broadcastable to the same shape. If choices is itself an
                 array (not recommended), then its outermost dimension (i.e., the one corresponding to choices.shape[0])
                  is taken as defining the “sequence”.

            out: array, optional

                If provided, the result will be inserted into this array. It should be of the appropriate shape and
                dtype. Note that out is always buffered if mode='raise'; use other modes for better performance.

            mode{‘raise’ (default), ‘wrap’, ‘clip’}, optional

                Specifies how indices outside [0, n-1] will be treated:

                        ‘raise’ : an exception is raised

                        ‘wrap’ : value becomes value mod n

                        ‘clip’ : values < 0 are mapped to 0, values > n-1 are mapped to n-1

        Returns
            merged_array: PhiTensor
                The merged result.

        Raises
            ValueError: shape mismatch
                If a and each choice array are not all broadcastable to the same shape.

        """

        if isinstance(choices, PhiTensor):
            if (self.data_subjects != choices.data_subjects).any():
                return self.gamma.choose(choices.gamma, mode=mode)
            else:
                result = self.child.choose(choices.child, mode=mode)
                output_ds = np.choose(self.child, choices.data_subjects)
        elif isinstance(choices, GammaTensor):
            return self.gamma.choose(choices, mode=mode)
        else:
            raise NotImplementedError(
                f"Object type: {type(choices)} This leads to a data leak or side channel attack"
            )

        return PhiTensor(
            child=result,
            data_subjects=output_ds,
            min_vals=lazyrepeatarray(
                data=choices.min_vals.data.min(), shape=result.shape
            ),
            max_vals=lazyrepeatarray(
                data=choices.max_vals.data.max(), shape=result.shape
            ),
        )

    def cumsum(
        self,
        axis: Optional[int] = None,
    ) -> PhiTensor:
        """
        Return the cumulative sum of the elements along a given axis.

        Parameters
            axis: int, optional
                Axis along which the cumulative sum is computed. The default (None) is to compute the cumsum over the
                flattened array.
        Returns
            cumsum_along_axis: PhiTensor
                A new array holding the result is returned. The result has the same size as input, and the same shape as
                 a if axis is not None or a is 1-d.
        """
        result = self.child.cumsum(axis=axis)
        num = np.ones_like(self.child).cumsum(axis=axis)
        return PhiTensor(
            child=result,
            data_subjects=self.data_subjects.cumsum(axis=axis),
            min_vals=lazyrepeatarray(
                data=(self.min_vals.data * num).min(), shape=result.shape
            ),
            max_vals=lazyrepeatarray(
                data=(self.max_vals.data * num).max(), shape=result.shape
            ),
        )

    def cumprod(
        self,
        axis: Optional[int] = None,
    ) -> PhiTensor:
        """
        Return the cumulative product of the elements along a given axis.

        Parameters
            axis: int, optional
                Axis along which the cumulative product is computed. The default (None) is to compute the cumprod over
                the flattened array.
        Returns
            cumprod_along_axis: PhiTensor
                A new array holding the result is returned. The result has the same size as input, and the same shape as
                 a if axis is not None or a is 1-d.
        """
        result = self.child.cumprod(axis=axis)
        num = np.ones_like(self.child).cumsum(axis=axis)
        if abs(self.max_vals.data) >= (self.min_vals.data):
            highest = abs(self.max_vals.data)
        else:
            highest = abs(self.min_vals.data)

        return PhiTensor(
            child=result,
            data_subjects=self.data_subjects.cumprod(axis=axis),
            min_vals=lazyrepeatarray(
                data=-((highest**num).max()), shape=result.shape
            ),
            max_vals=lazyrepeatarray(data=(highest**num).max(), shape=result.shape),
        )

    def prod(self, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> PhiTensor:
        """
        Return the product of array elements over a given axis.

        Parameters
            axis: None or int or tuple of ints, optional
                Axis or axes along which a product is performed.
                The default, axis=None, will calculate the product of all the elements in the input array.
                If axis is negative it counts from the last to the first axis.

                If axis is a tuple of ints, a product is performed on all of the axes specified in the tuple instead of
                a single axis or all the axes as before.

            keepdims: bool, optional
                If this is set to True, the axes which are reduced are left in the result as dimensions with size one.
                With this option, the result will broadcast correctly against the input array.
                If the default value is passed, then keepdims will not be passed through to the prod method of
                sub-classes of ndarray, however any non-default value will be. If the sub-class’ method does not
                implement keepdims any exceptions will be raised.

            initial: scalar, optional
                The starting value for this product. See reduce for details.

            where: array_like of bool, optional
                Elements to include in the product. See reduce for details.
        """
        result = self.child.prod(axis=axis)
        return PhiTensor(
            child=result,
            data_subjects=self.data_subjects.prod(axis),
            min_vals=lazyrepeatarray(
                data=self.min_vals.data ** (self.child.size / result.size),
                shape=result.shape,
            ),
            max_vals=lazyrepeatarray(
                data=self.max_vals.data ** (self.child.size / result.size),
                shape=result.shape,
            ),
        )

    def __floordiv__(self, other: Any) -> Union[PhiTensor, GammaTensor]:
        """
        return self // value.
        """
        if isinstance(other, PhiTensor):
            if np.array(self.data_subjects != other.data_subjects).all():
                return self.gamma // other.gamma
            else:
                min_min = self.min_vals.data // other.min_vals.data
                min_max = self.min_vals.data // other.max_vals.data
                max_min = self.max_vals.data // other.min_vals.data
                max_max = self.max_vals.data // other.max_vals.data

                _min_vals = np.min([min_min, min_max, max_min, max_max], axis=0)  # type: ignore
                _max_vals = np.max([min_min, min_max, max_min, max_max], axis=0)  # type: ignore

                return PhiTensor(
                    child=self.child // other.child,
                    data_subjects=self.data_subjects,
                    min_vals=lazyrepeatarray(data=_min_vals, shape=self.shape),
                    max_vals=lazyrepeatarray(data=_max_vals, shape=self.shape),
                )
        elif isinstance(other, GammaTensor):
            return self.gamma // other
        elif is_acceptable_simple_type(other):
            return PhiTensor(
                child=self.child // other,
                data_subjects=self.data_subjects,
                min_vals=lazyrepeatarray(
                    data=self.min_vals.data // other, shape=self.min_vals.shape
                ),
                max_vals=lazyrepeatarray(
                    data=self.max_vals.data // other, shape=self.max_vals.shape
                ),
            )
        else:
            raise NotImplementedError(
                f"floordiv not supported between PhiTensor & {type(other)}"
            )

    def trace(self, offset: int = 0, axis1: int = 0, axis2: int = 1) -> PhiTensor:
        """
        Return the sum along diagonals of the array.

        If a is 2-D, the sum along its diagonal with the given offset is returned, i.e., the sum of elements
        a[i,i+offset] for all i.

        If a has more than two dimensions, then the axes specified by axis1 and axis2 are used to determine the 2-D
        sub-arrays whose traces are returned. The shape of the resulting array is the same as that of a with axis1 and
        axis2 removed.

        Parameters

            offset: int, optional
                Offset of the diagonal from the main diagonal. Can be both positive and negative. Defaults to 0.

            axis1, axis2: int, optional
                Axes to be used as the first and second axis of the 2-D sub-arrays from which the diagonals should be
                taken. Defaults are the first two axes of a.

        Returns

            sum_along_diagonals: PhiTensor
                If a is 2-D, the sum along the diagonal is returned.
                If a has larger dimensions, then an array of sums along diagonals is returned.
        """
        result = self.child.trace(offset, axis1, axis2)

        # This is potentially expensive
        num = np.ones_like(self.child).trace(offset, axis1, axis2)
        return PhiTensor(
            child=result,
            data_subjects=self.data_subjects.trace(offset, axis1, axis2),
            min_vals=lazyrepeatarray(data=self.min_vals.data * num, shape=result.shape),
            max_vals=lazyrepeatarray(data=self.max_vals.data * num, shape=result.shape),
        )

    def diagonal(self, offset: int = 0, axis1: int = 0, axis2: int = 1) -> PhiTensor:

        result = self.child.diagonal(offset, axis1, axis2)

        return PhiTensor(
            child=result,
            data_subjects=self.data_subjects.diagonal(offset, axis1, axis2),
            min_vals=lazyrepeatarray(data=self.min_vals.data, shape=result.shape),
            max_vals=lazyrepeatarray(data=self.max_vals.data, shape=result.shape),
        )

    def _object2bytes(self) -> bytes:
        schema = get_capnp_schema(schema_file="phi_tensor.capnp")

        pt_struct: CapnpModule = schema.PT  # type: ignore
        pt_msg = pt_struct.new_message()
        # this is how we dispatch correct deserialization of bytes
        pt_msg.magicHeader = serde_magic_header(type(self))

        if isinstance(self.child, np.ndarray) or np.isscalar(self.child):
            chunk_bytes(capnp_serialize(np.array(self.child), to_bytes=True), "child", pt_msg)  # type: ignore
            pt_msg.isNumpy = True
        else:
            chunk_bytes(serialize(self.child, to_bytes=True), "child", pt_msg)  # type: ignore
            pt_msg.isNumpy = False

        pt_msg.minVals = serialize(self.min_vals, to_bytes=True)
        pt_msg.maxVals = serialize(self.max_vals, to_bytes=True)
        chunk_bytes(
            capnp_serialize(dslarraytonumpyutf8(self.data_subjects), to_bytes=True),
            "dataSubjects",
            pt_msg,
        )
        # to pack or not to pack?
        # to_bytes = pt_msg.to_bytes()

        return pt_msg.to_bytes_packed()

    @staticmethod
    def _bytes2object(buf: bytes) -> PhiTensor:
        schema = get_capnp_schema(schema_file="phi_tensor.capnp")
        pt_struct: CapnpModule = schema.PT  # type: ignore
        # https://stackoverflow.com/questions/48458839/capnproto-maximum-filesize
        MAX_TRAVERSAL_LIMIT = 2**64 - 1
        # to pack or not to pack?
        # pt_msg = pt_struct.from_bytes(buf, traversal_limit_in_words=2 ** 64 - 1)
        pt_msg = pt_struct.from_bytes_packed(
            buf, traversal_limit_in_words=MAX_TRAVERSAL_LIMIT
        )

        if pt_msg.isNumpy:
            child = capnp_deserialize(combine_bytes(pt_msg.child), from_bytes=True)
        else:
            child = deserialize(combine_bytes(pt_msg.child), from_bytes=True)

        min_vals = deserialize(pt_msg.minVals, from_bytes=True)
        max_vals = deserialize(pt_msg.maxVals, from_bytes=True)
        data_subjects = numpyutf8todslarray(
            capnp_deserialize(combine_bytes(pt_msg.dataSubjects), from_bytes=True)
        )

        return PhiTensor(
            child=child,
            min_vals=min_vals,
            max_vals=max_vals,
            data_subjects=data_subjects,
        )
