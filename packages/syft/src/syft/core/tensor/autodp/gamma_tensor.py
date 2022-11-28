# future
from __future__ import annotations

# stdlib
from collections import deque
from typing import Any
from typing import Callable
from typing import Deque
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import TYPE_CHECKING
from typing import Tuple
from typing import Union

# third party
import flax
import jax
from jax import numpy as jnp
import numpy as np
from numpy.random import randint
from numpy.typing import ArrayLike
from numpy.typing import NDArray
from scipy.optimize import shgo

# relative
from .... import lib
from ....ast.klass import pointerize_args_and_kwargs
from ....core.node.common.action.get_or_set_property_action import (
    GetOrSetPropertyAction,
)
from ....core.node.common.action.get_or_set_property_action import PropertyActions
from ....lib.numpy.array import capnp_deserialize
from ....lib.numpy.array import capnp_serialize
from ....lib.python.util import upcast
from ....util import inherit_tags
from ...adp.data_subject_ledger import DataSubjectLedger

# from ...adp.data_subject_list import DataSubjectList
from ...adp.data_subject_list import DataSubjectArray
from ...adp.data_subject_list import dslarraytonumpyutf8
from ...adp.data_subject_list import numpyutf8todslarray
from ...adp.vectorized_publish import publish
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
from ..config import DEFAULT_INT_NUMPY_TYPE
from ..fixed_precision_tensor import FixedPrecisionTensor
from ..lazy_repeat_array import compute_min_max
from ..lazy_repeat_array import lazyrepeatarray
from ..passthrough import PassthroughTensor  # type: ignore
from ..passthrough import SupportedChainType  # type: ignore
from ..passthrough import is_acceptable_simple_type  # type: ignore
from ..smpc import utils
from ..smpc.mpc_tensor import MPCTensor
from ..smpc.utils import TYPE_TO_RING_SIZE
from ..util import implements
from .gamma_tensor_ops import GAMMA_TENSOR_OP

if TYPE_CHECKING:
    # stdlib
    from dataclasses import dataclass
else:
    # third party
    from flax.struct import dataclass


INPLACE_OPS = {"resize", "sort"}


@serializable(recursive_serde=True)
class TensorWrappedGammaTensorPointer(Pointer):
    __name__ = "TensorWrappedGammaTensorPointer"
    __module__ = "syft.core.tensor.autodp.gamma_tensor"
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
        data_subjects: DataSubjectArray,
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
            * (self.max_vals.to_numpy() - self.min_vals.to_numpy())
            + self.min_vals.to_numpy()
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
        result = TensorWrappedGammaTensorPointer(
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

        if isinstance(other, TensorWrappedGammaTensorPointer):
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
                f"Invalid Type for TensorWrappedGammaTensorPointer:{type(other)}"
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

    @staticmethod
    def _apply_op(
        self: TensorWrappedGammaTensorPointer,
        other: Union[
            TensorWrappedGammaTensorPointer, MPCTensor, int, float, np.ndarray
        ],
        op_str: str,
    ) -> Union[MPCTensor, TensorWrappedGammaTensorPointer]:
        """Performs the operation based on op_str

        Args:
            other (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]): second operand.

        Returns:
            Tuple[MPCTensor,Union[MPCTensor,int,float,np.ndarray]] : Result of the operation
        """
        # relative
        from ..autodp.phi_tensor import TensorWrappedPhiTensorPointer

        if isinstance(other, TensorWrappedPhiTensorPointer):
            other = other.gamma

        if (
            isinstance(other, TensorWrappedGammaTensorPointer)
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

        return self._apply_tensor_op(other=other, op_str=op_str)

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
                # relative
                from .phi_tensor import TensorWrappedPhiTensorPointer

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
                                TensorWrappedGammaTensorPointer,
                                TensorWrappedPhiTensorPointer,
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
                                TensorWrappedGammaTensorPointer,
                                TensorWrappedPhiTensorPointer,
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

        result = TensorWrappedGammaTensorPointer(
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

        return result

    def copy(self, *args: Any, **kwargs: Any) -> TensorWrappedGammaTensorPointer:
        return self._apply_self_tensor_op("copy", *args, **kwargs)

    def __add__(
        self,
        other: Union[
            TensorWrappedGammaTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """Apply the "add" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedGammaTensorPointer._apply_op(self, other, "__add__")

    def __sub__(
        self,
        other: Union[
            TensorWrappedGammaTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """Apply the "sub" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedGammaTensorPointer._apply_op(self, other, "__sub__")

    def __mul__(
        self,
        other: Union[
            TensorWrappedGammaTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """Apply the "mul" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedGammaTensorPointer._apply_op(self, other, "__mul__")

    def __matmul__(
        self,
        other: Union[
            TensorWrappedGammaTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """Apply the "matmul" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedGammaTensorPointer._apply_op(self, other, "__matmul__")

    def __rmatmul__(
        self,
        other: Union[
            TensorWrappedGammaTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """Apply the "rmatmul" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedGammaTensorPointer._apply_op(self, other, "__rmatmul__")

    def __lt__(
        self,
        other: Union[
            TensorWrappedGammaTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """Apply the "lt" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedGammaTensorPointer._apply_op(self, other, "__lt__")

    def __gt__(
        self,
        other: Union[
            TensorWrappedGammaTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """Apply the "gt" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedGammaTensorPointer._apply_op(self, other, "__gt__")

    def __ge__(
        self,
        other: Union[
            TensorWrappedGammaTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """Apply the "ge" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedGammaTensorPointer._apply_op(self, other, "__ge__")

    def __le__(
        self,
        other: Union[
            TensorWrappedGammaTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """Apply the "le" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedGammaTensorPointer._apply_op(self, other, "__le__")

    def __eq__(  # type: ignore
        self,
        other: Union[
            TensorWrappedGammaTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """Apply the "eq" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedGammaTensorPointer._apply_op(self, other, "__eq__")

    def __ne__(  # type: ignore
        self,
        other: Union[
            TensorWrappedGammaTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """Apply the "ne" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedGammaTensorPointer._apply_op(self, other, "__ne__")

    def concatenate(
        self,
        other: TensorWrappedGammaTensorPointer,
        *args: Any,
        **kwargs: Any,
    ) -> MPCTensor:
        """Apply the "concatenate" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.


        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        if not isinstance(other, TensorWrappedGammaTensorPointer):
            raise ValueError(
                f"Concatenate works only for TensorWrappedGammaTensorPointer got type: {type(other)}"
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

    def __truediv__(
        self,
        other: Union[
            TensorWrappedGammaTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """Apply the "truediv" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedGammaTensorPointer._apply_op(self, other, "__truediv__")

    def __mod__(
        self,
        other: Union[
            TensorWrappedGammaTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        return TensorWrappedGammaTensorPointer._apply_op(self, other, "__mod__")

    def __and__(
        self,
        other: Union[
            TensorWrappedGammaTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """Apply the "and" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedGammaTensorPointer._apply_op(self, other, "__and__")

    def __or__(
        self,
        other: Union[
            TensorWrappedGammaTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """Apply the "or" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedGammaTensorPointer._apply_op(self, other, "__or__")

    def __floordiv__(
        self,
        other: Union[
            TensorWrappedGammaTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """Apply the "floordiv" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedGammaTensorPointer._apply_op(self, other, "__floordiv__")

    def __divmod__(
        self,
        other: Union[
            TensorWrappedGammaTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Tuple[
        Union[TensorWrappedGammaTensorPointer, MPCTensor],
        Union[TensorWrappedGammaTensorPointer, MPCTensor],
    ]:
        """Apply the "divmod" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        return self.divmod(other)

    def divmod(
        self,
        other: Union[
            TensorWrappedGammaTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Tuple[
        Union[TensorWrappedGammaTensorPointer, MPCTensor],
        Union[TensorWrappedGammaTensorPointer, MPCTensor],
    ]:
        """Apply the "divmod" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedGammaTensorPointer._apply_op(
            self, other, "__floordiv__"
        ), TensorWrappedGammaTensorPointer._apply_op(self, other, "__mod__")

    def sum(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
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
        return self._apply_self_tensor_op("sum", *args, **kwargs)

    def ptp(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """Apply the "ptp" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        return self._apply_self_tensor_op("ptp", *args, **kwargs)

    def __lshift__(
        self,
        other: Union[
            TensorWrappedGammaTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """Apply the "lshift" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedGammaTensorPointer._apply_op(self, other, "__lshift__")

    def argmax(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """Apply the "argmax" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        return self._apply_self_tensor_op("argmax", *args, **kwargs)

    def __rshift__(
        self,
        other: Union[
            TensorWrappedGammaTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """Apply the "rshift" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedGammaTensorPointer._apply_op(self, other, "__rshift__")

    def argmin(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """Apply the "argmin" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        return self._apply_self_tensor_op("argmin", *args, **kwargs)

    def __abs__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """Apply the "abs" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        return self._apply_self_tensor_op("__abs__", *args, **kwargs)

    def all(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """Apply the "all" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        return self._apply_self_tensor_op("all", *args, **kwargs)

    def any(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """Apply the "any" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        return self._apply_self_tensor_op("any", *args, **kwargs)

    def round(self, *args: Any, **kwargs: Any) -> TensorWrappedGammaTensorPointer:
        return self._apply_self_tensor_op("round", *args, **kwargs)

    def __round__(self, *args: Any, **kwargs: Any) -> TensorWrappedGammaTensorPointer:
        return self.round(*args, **kwargs)

    def __pos__(self) -> TensorWrappedGammaTensorPointer:
        """Apply the pos (+) operator  on self.

        Returns:
            Union[TensorWrappedGammaTensorPointer] : Result of the operation.
        """
        return self._apply_self_tensor_op(op_str="__pos__")

    def var(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
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

    def cumsum(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """ "
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

    def cumprod(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
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

    def prod(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
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

    def __xor__(
        self,
        other: Union[
            TensorWrappedGammaTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """Apply the "xor" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedGammaTensorPointer._apply_op(self, other, "__xor__")

    def __pow__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
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

    def mean(self, *args: Any, **kwargs: Any) -> TensorWrappedGammaTensorPointer:
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

    def std(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
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
        data_subjects = np.array(self.data_subjects).std(*args, **kwargs)  # type: ignore
        result = TensorWrappedGammaTensorPointer(
            data_subjects=data_subjects,
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

    def trace(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
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

            Union[TensorWrappedPhiTensorPointer,MPCTensor] : Result of the operation.
                If a is 2-D, the sum along the diagonal is returned.
                If a has larger dimensions, then an array of sums along diagonals is returned.

        """
        return self._apply_self_tensor_op("trace", *args, **kwargs)

    def sort(self, *args: Any, **kwargs: Any) -> TensorWrappedGammaTensorPointer:
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

    def argsort(self, *args: Any, **kwargs: Any) -> TensorWrappedGammaTensorPointer:
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

    def min(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
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

    def max(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
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

    def compress(
        self, *args: Any, **kwargs: Any
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """
        Return selected slices of an array along given axis.

        When working along a given axis, a slice along that axis is returned in output for each index
        where condition evaluates to True. When working on a 1-D array, compress is equivalent to extract.

        Parameters
            condition: 1-D array of bools
            Array that selects which entries to return. If len(condition) is less than the size of
            a along the given axis,then output is truncated to the length of the condition array.

            axis: int, optional
            Axis along which to take slices. If None (default), work on the flattened array.

        Returns:
            compressed_array: PhiTensor
            A copy of a without the slices along axis for which condition is false.
        """
        return self._apply_self_tensor_op("compress", *args, **kwargs)

    def squeeze(
        self, *args: Any, **kwargs: Any
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
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

    def __getitem__(
        self, key: Union[int, bool, slice]
    ) -> TensorWrappedGammaTensorPointer:
        """Return self[key].
        Args:
            y (Union[int,bool,slice]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer] : Result of the operation.
        """
        return self._apply_self_tensor_op("__getitem__", key)

    def zeros_like(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """Apply the "zeros_like" operation on "self"

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        return self._apply_self_tensor_op("zeros_like", *args, **kwargs)

    def ones_like(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """Apply the "ones_like" operation on "self"

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        return self._apply_self_tensor_op("ones_like", *args, **kwargs)

    def transpose(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """
        Reverse or permute the axes of an array; returns the modified array.

        Returns
            p: ndarray
                array with its axes permuted. A view is returned whenever possible.
        """

        return self._apply_self_tensor_op("transpose", *args, **kwargs)

    def resize(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:

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

    def reshape(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:

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

    def repeat(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """Apply the repeat" operation

        Args:
            y (Union[TensorWrappedPhiTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedPhiTensorPointer,MPCTensor] : Result of the operation.
        """
        return self._apply_self_tensor_op("repeat", *args, **kwargs)

    def diagonal(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """
        Return specified diagonals.
        If a is 2-D, returns the diagonal of a with the given offset, i.e., the collection of elements
        of the form a[i, i+offset].

        If a has more than two dimensions, then the axes specified by axis1 and axis are used to determine
        the 2-D sub-array whose diagonal is returned.  The shape of the resulting array can be determined by
        removing axis1 and axis2 and appending an index to the right equal to the size of the resulting diagonals.

        Parameters

            offset: int, optional
                Offset of the diagonal from the main diagonal.  Can be positive or negative.
                Defaults to main diagonal (0).
            axis1, axis2: int, optional
                Axis to be used as the first axis of the 2-D sub-arrays from which the diagonals should be taken.
                Defaults are the first two axes of a.

        Returns
            array_of_diagonals : Union[TensorWrappedPhiTensorPointer,MPCTensor]
                If a is 2-D, then a 1-D array containing the diagonal and of the same type as a is returned unless
                a is a matrix, in which case
                a 1-D array rather than a (2-D) matrix is returned in order to maintain backward compatibility.

                If a.ndim > 2, then the dimensions specified by axis1 and axis2 are removed, and a new axis
                inserted at the end corresponding to the diagonal.
        """
        return self._apply_self_tensor_op("diagonal", *args, **kwargs)

    def flatten(
        self, *args: Any, **kwargs: Any
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """
        Return a copy of the array collapsed into one dimension.

        Parameters
            order: {‘C’, ‘F’, ‘A’, ‘K’}, optional
            ‘C’ means to flatten in row-major (C-style) order.
            ‘F’ means to flatten in column-major (Fortran- style) order.
            ‘A’ means to flatten in column-major order if a is Fortran contiguous in memory, row-major order otherwise.
            ‘K’ means to flatten a in the order the elements occur in memory. The default is ‘C’.

        Returns
            y: PhiTensor
                A copy of the input array, flattened to one dimension.
        """
        return self._apply_self_tensor_op("flatten", *args, **kwargs)

    def ravel(
        self, *args: Any, **kwargs: Any
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """
        Return a contiguous flattened array.

        A 1-D array, containing the elements of the input, is returned. A copy is made only if needed.

        As of NumPy 1.10, the returned array will have the same type as the input array.
        (for example, a masked array will be returned for a masked array input)
        Parameters
            order: {‘C’,’F’, ‘A’, ‘K’}, optional
            The elements of a are read using this index order.
            ‘C’ means to index the elements in row-major,
            C-style order, with the last axis index changing fastest, back to the first axis index changing slowest.
            ‘F’ means to index the elements in column-major, Fortran-style order, with the first index changing fastest,
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
                Note that matrices are special cased for backward compatibility,
                if a is a matrix, then y is a 1-D ndarray.
        """
        return self._apply_self_tensor_op("ravel", *args, **kwargs)

    def take(
        self, *args: Any, **kwargs: Any
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
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

    def clip(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
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
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
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
    def T(self) -> TensorWrappedGammaTensorPointer:
        # We always maintain a Tensor hierarchy Tensor ---> PT--> Actual Data
        attr_path_and_name = "syft.core.tensor.tensor.Tensor.T"

        result = TensorWrappedGammaTensorPointer(
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

    def to_local_object_without_private_data_child(self) -> GammaTensor:
        """Convert this pointer into a partial version of the GammaTensor but without
        any of the private data therein."""
        # relative
        from ..tensor import Tensor

        public_shape = getattr(self, "public_shape", None)
        public_dtype = getattr(self, "public_dtype", None)
        return Tensor(
            child=GammaTensor(
                child=FixedPrecisionTensor(value=None),
                data_subjects=self.data_subjects,
                min_vals=self.min_vals,  # type: ignore
                max_vals=self.max_vals,  # type: ignore
            ),
            public_shape=public_shape,
            public_dtype=public_dtype,
        )


@implements(TensorWrappedGammaTensorPointer, np.zeros_like)
def zeros_like(
    tensor: TensorWrappedGammaTensorPointer,
    *args: Any,
    **kwargs: Any,
) -> TensorWrappedGammaTensorPointer:
    return tensor.zeros_like(*args, **kwargs)


@implements(TensorWrappedGammaTensorPointer, np.ones_like)
def ones_like(
    tensor: TensorWrappedGammaTensorPointer,
    *args: Any,
    **kwargs: Any,
) -> TensorWrappedGammaTensorPointer:
    return tensor.ones_like(*args, **kwargs)


def create_lookup_tables(dictionary: dict) -> Tuple[List[str], dict, List[dict]]:
    index2key: List = [str(x) for x in dictionary.keys()]
    key2index: dict = {key: i for i, key in enumerate(index2key)}
    # Note this maps to GammaTensor, not to GammaTensor.child as name may imply
    index2values: List = [dictionary[i] for i in index2key]

    return index2key, key2index, index2values


def create_new_lookup_tables(
    dictionary: dict,
) -> Tuple[Deque[str], dict, Deque[dict], Deque[int]]:
    index2key: Deque = deque()
    key2index: dict = {}
    index2values: Deque = (
        deque()
    )  # Note this maps to GammaTensor, not to GammaTensor.child as name may imply
    index2size: Deque = deque()
    for index, key in enumerate(dictionary.keys()):
        key = str(key)
        index2key.append(key)
        key2index[key] = index
        index2values.append(dictionary[key])
        index2size.append(len(dictionary[key]))

    return index2key, key2index, index2values, index2size


# def no_op(x: GammaTensor) -> GammaTensor:
#     """A Private input will be initialized with this function.
#     Whenever you manipulate a private input (i.e. add it to another private tensor),
#     the result will have a different function. Thus we can check to see if the f
#     """
#     res = x
#     if isinstance(x, GammaTensor) and isinstance(x.data_subjects, np.ndarray):
#         res = GammaTensor(
#             child=x.child,
#             data_subjects=np.zeros_like(x.data_subjects, np.int64),
#             min_vals=x.min_vals,
#             max_vals=x.max_vals,
#             func_str=x.func_str,
#             sources=GammaTensor.convert_dsl(x.sources),
#         )
#     return res


def jax2numpy(value: jnp.array, dtype: np.dtype) -> np.array:
    # are we incurring copying here?
    return np.asarray(value, dtype=dtype)


def numpy2jax(value: np.array, dtype: np.dtype) -> jnp.array:
    return jnp.asarray(value, dtype=dtype)


# ATTENTION: Shouldn't this be a subclass of some kind of base tensor so all the numpy
# methods and properties don't need to be re-implemented on it?
@dataclass
@serializable(capnp_bytes=True)
class GammaTensor:
    """
    A differential privacy tensor that contains data belonging to atleast 2 or more unique data subjects.

    Attributes:
        child: jnp.array
            The private data itself.
        data_subjects: DataSubjectArray
            (DP Metadata) A custom NumPy class that keeps track of which data subjects contribute which datapoints in
            this tensor.
        min_vals: lazyrepeatarray
            (DP Metadata) A custom class that keeps track of (data-independent) minimum values for this tensor.
        max_vals: lazyrepeatarray
            (DP Metadata) A custom class that keeps track of (data-independent) maximum values for this tensor.
        func_str: str
            A string that will determine which function was used to build the current tensor.
        is_linear: bool
            Whether the "func_str" for this tensor is a linear query or not. This impacts the epsilon calculations
            when publishing.
        sources: dict
            A dictionary containing all the Tensors, integers, etc that were used to create this tensor.
            It maps an integer to each input object.
        id: int
            A 32-bit integer that is used when this GammaTensor needs to be added to the "sources" dictionary.

    Methods:
        All efforts were made to make this tensor's API as similar to the NumPy API as possible.
        Special, unique methods are listed below:

        reconstruct(sources: Optional[dict]):
            rebuilds the tensor from the sources dictionary provided, or from the current self.sources.
            This is exclusively used when adding DP Noise, if the data scientist doesn't have enough privacy budget to
            use one of the input tensors, thus requiring that tensor's data to be removed from the computation.

        swap_state(sources: Optional[Dict]):
            calls reconstruct() and populates the rest of the GammaTensor's attributes based on the current tensor.
            Used exclusively when adding DP Noise.



        decode():
            occasionally the use of a FixedPrecisionTensor (FPT) is needed during SMPC[1]. This helps convert back from
            FPT to regular numpy/jax arrays.

            (https://en.wikipedia.org/wiki/Secure_multi-party_computation)





    """

    PointerClassOverride = TensorWrappedGammaTensorPointer

    child: jnp.array
    data_subjects: np.ndarray
    min_vals: Union[lazyrepeatarray, np.ndarray] = flax.struct.field(pytree_node=False)
    max_vals: Union[lazyrepeatarray, np.ndarray] = flax.struct.field(pytree_node=False)
    is_linear: bool = True
    func_str: str = flax.struct.field(
        pytree_node=False, default_factory=lambda: GAMMA_TENSOR_OP.NOOP.value
    )
    id: str = flax.struct.field(
        pytree_node=False, default_factory=lambda: str(randint(0, 2**31 - 1))
    )  # TODO: Need to check if there are any scenarios where this is not secure
    sources: dict = flax.struct.field(pytree_node=False, default_factory=dict)

    def __post_init__(
        self,
    ) -> None:  # Might not serve any purpose anymore, since state trees are updated during ops
        if (
            self.sources
            and len(self.sources) == 0
            and self.func_str != GAMMA_TENSOR_OP.NOOP.value
        ):
            self.sources[self.id] = self

        if isinstance(self.min_vals, lazyrepeatarray):
            if self.min_vals.data.size != 1:
                self.min_vals.data = self.min_vals.data.min()
            if self.max_vals.data.size != 1:
                self.max_vals.data = self.max_vals.data.max()

    def decode(self) -> np.ndarray:
        if isinstance(self.child, FixedPrecisionTensor):
            return self.child.decode()
        else:
            return self.child

    @property
    def proxy_public_kwargs(self) -> Dict[str, Any]:
        return {
            "min_vals": self.min_vals,
            "max_vals": self.max_vals,
            "data_subjects": self.data_subjects,
        }

    def reconstruct(self, state: Optional[Dict] = None) -> GammaTensor:
        if self.func_str == GAMMA_TENSOR_OP.NOOP.value:
            # ATTENTION:
            # during publish we attempt to remove nodes if the we exceed budget
            # if we call swap_state on a terminal Tensor we need to replace the
            # child with zeros not the current level tensors child which is
            # not zeroed yet
            if state is not None and len(state.keys()) == 1:
                # return the swapped state child to replace
                try:
                    return list(state.values())[0].child
                except Exception as e:
                    print("Something bad has happened in reconstruct", state)
                    raise e
            else:
                # ATTENTION:
                # can we have a terminal no_op tensor with multiple state keys?
                pass
            return self.child
        else:
            # relative
            from .gamma_functions import GAMMA_FUNC_MAPPER

            jax_op = GAMMA_FUNC_MAPPER[GAMMA_TENSOR_OP(self.func_str)]
            return jax_op(state if state is not None else self.sources)

    def swap_state(self, state: dict) -> GammaTensor:
        return GammaTensor(
            child=self.reconstruct(state),
            data_subjects=self.data_subjects,
            min_vals=self.min_vals,
            max_vals=self.max_vals,
            sources=state,
            func_str=self.func_str,
            is_linear=self.is_linear,
        )

    @property
    def size(self) -> int:
        if (
            isinstance(self.child, float)
            or isinstance(self.child, int)
            or isinstance(self.child, bool)
        ):
            return 1

        if hasattr(self.child, "size"):
            return self.child.size
        elif hasattr(self.child, "shape"):
            return np.prod(self.child.shape)

        raise Exception(f"{type(self)} has no attribute size.")

    # def all(self) -> bool:
    #     if hasattr(self.child, "all"):
    #         return self.child.all()
    #     elif isinstance(self.child, Iterable):
    #         return all(self.child)
    #     return bool(self.child)

    def __add__(self, other: Any) -> GammaTensor:
        # relative
        from .phi_tensor import PhiTensor

        output_state = dict()
        # Add this tensor to the chain
        output_state[self.id] = self

        if isinstance(other, PhiTensor):
            other = other.gamma

        if isinstance(other, GammaTensor):
            output_state[other.id] = other

            child = self.child + other.child
            min_val = self.min_vals + other.min_vals
            max_val = self.max_vals + other.max_vals
            output_ds = self.data_subjects + other.data_subjects

        else:
            output_state[np.random.randint(low=0, high=2**31 - 1)] = other

            child = self.child + other
            min_val = self.min_vals + other
            max_val = self.max_vals + other
            output_ds = self.data_subjects

        return GammaTensor(
            child=child,
            data_subjects=output_ds,
            min_vals=min_val,
            max_vals=max_val,
            func_str=GAMMA_TENSOR_OP.ADD.value,
            sources=output_state,
        )

    def __mod__(self, other: Any) -> GammaTensor:
        # relative
        from .phi_tensor import PhiTensor

        output_state = dict()
        # Add this tensor to the chain
        output_state[self.id] = self

        if isinstance(other, PhiTensor):
            other = other.gamma

        if isinstance(other, GammaTensor):
            output_state[other.id] = other

            child = self.child % other.child
            max_vals = lazyrepeatarray(
                data=max(0, other.max_vals.data), shape=self.child.shape
            )
            min_vals = lazyrepeatarray(
                data=min(0, other.min_vals.data), shape=self.child.shape
            )
            output_ds = self.data_subjects + other.data_subjects

        elif is_acceptable_simple_type(other):
            output_state[np.random.randint(low=0, high=2**31 - 1)] = other
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

            child = self.child % other
            output_ds = self.data_subjects
        else:
            print("Type is unsupported:" + str(type(other)))
            raise NotImplementedError

        return GammaTensor(
            child=child,
            data_subjects=output_ds,
            min_vals=min_vals,
            max_vals=max_vals,
            func_str=GAMMA_TENSOR_OP.MOD.value,
            sources=output_state,
        )

    def __rtruediv__(self, other: SupportedChainType) -> GammaTensor:

        if is_acceptable_simple_type(other):
            return GammaTensor(
                child=(1 / self.child) * other,
                min_vals=(1 / self.min_vals) * other,
                max_vals=(1 / self.max_vals) * other,
                data_subjects=(1 / self.data_subjects) * other,
            )
        else:
            print("Type is unsupported:" + str(type(other)))
            raise NotImplementedError

    def __sub__(self, other: Any) -> GammaTensor:
        # relative
        from .phi_tensor import PhiTensor

        output_state = dict()
        # Add this tensor to the chain
        output_state[self.id] = self

        if isinstance(other, PhiTensor):
            other = other.gamma

        if isinstance(other, GammaTensor):
            output_state[other.id] = other

            child = self.child - other.child
            min_min = self.min_vals.data - other.min_vals.data
            min_max = self.min_vals.data - other.max_vals.data
            max_min = self.max_vals.data - other.min_vals.data
            max_max = self.max_vals.data - other.max_vals.data
            _min_val = np.minimum.reduce([min_min, min_max, max_min, max_max])
            _max_val = np.maximum.reduce([min_min, min_max, max_min, max_max])
            min_val = lazyrepeatarray(data=_min_val, shape=self.shape)
            max_val = lazyrepeatarray(data=_max_val, shape=self.shape)

            output_ds = self.data_subjects - other.data_subjects

        else:
            child = self.child - other
            min_val = self.min_vals - other
            max_val = self.max_vals - other
            output_ds = self.data_subjects
            output_state[np.random.randint(low=0, high=2**31 - 1)] = other

        return GammaTensor(
            child=child,
            data_subjects=output_ds,
            min_vals=min_val,
            max_vals=max_val,
            func_str=GAMMA_TENSOR_OP.SUBTRACT.value,
            sources=output_state,
        )

    def __mul__(self, other: Any) -> GammaTensor:
        # relative
        from .phi_tensor import PhiTensor

        output_state = dict()
        # Add this tensor to the chain
        output_state[self.id] = self

        if isinstance(other, PhiTensor):
            other = other.gamma

        if isinstance(other, GammaTensor):
            output_state[other.id] = other
            child = self.child * other.child
            min_min = self.min_vals.data * other.min_vals.data
            min_max = self.min_vals.data * other.max_vals.data
            max_min = self.max_vals.data * other.min_vals.data
            max_max = self.max_vals.data * other.max_vals.data
            _min_val = np.array(np.min([min_min, min_max, max_min, max_max], axis=0))  # type: ignore
            _max_val = np.array(np.max([min_min, min_max, max_min, max_max], axis=0))  # type: ignore
            output_ds = self.data_subjects * other.data_subjects

        else:
            child = self.child * other
            min_min = self.min_vals.data * other
            min_max = self.min_vals.data * other
            max_min = self.max_vals.data * other
            max_max = self.max_vals.data * other
            _min_val = np.array(np.min([min_min, min_max, max_min, max_max], axis=0))  # type: ignore
            _max_val = np.array(np.max([min_min, min_max, max_min, max_max], axis=0))  # type: ignore
            output_ds = self.data_subjects * other
            output_state[np.random.randint(low=0, high=2**31 - 1)] = other

        min_val = self.min_vals.copy()
        min_val.data = _min_val
        max_val = self.max_vals.copy()
        max_val.data = _max_val

        return GammaTensor(
            child=child,
            data_subjects=output_ds,
            min_vals=min_val,
            max_vals=max_val,
            func_str=GAMMA_TENSOR_OP.MULTIPLY.value,
            sources=output_state,
        )

    def __truediv__(self, other: Any) -> GammaTensor:
        # relative
        from .phi_tensor import PhiTensor

        output_state = dict()
        # Add this tensor to the chain
        output_state[self.id] = self

        if isinstance(other, PhiTensor):
            other = other.gamma

        if isinstance(other, GammaTensor):
            output_state[other.id] = other
            child = self.child / other.child
            min_min = self.min_vals.data / other.min_vals.data
            min_max = self.min_vals.data / other.max_vals.data
            max_min = self.max_vals.data / other.min_vals.data
            max_max = self.max_vals.data / other.max_vals.data
            _min_val = np.array(np.min([min_min, min_max, max_min, max_max], axis=0))  # type: ignore
            _max_val = np.array(np.max([min_min, min_max, max_min, max_max], axis=0))  # type: ignore
            output_ds = self.data_subjects * other.data_subjects

        else:
            child = self.child / other
            min_min = self.min_vals.data / other
            min_max = self.min_vals.data / other
            max_min = self.max_vals.data / other
            max_max = self.max_vals.data / other
            _min_val = np.array(np.min([min_min, min_max, max_min, max_max], axis=0))  # type: ignore
            _max_val = np.array(np.max([min_min, min_max, max_min, max_max], axis=0))  # type: ignore
            output_ds = self.data_subjects
            output_state[np.random.randint(low=0, high=2**31 - 1)] = other

        min_val = self.min_vals.copy()
        min_val.data = _min_val
        max_val = self.max_vals.copy()
        max_val.data = _max_val

        return GammaTensor(
            child=child,
            data_subjects=output_ds,
            min_vals=min_val,
            max_vals=max_val,
            func_str=GAMMA_TENSOR_OP.TRUE_DIVIDE.value,
            sources=output_state,
        )

    def __divmod__(self, other: Any) -> Tuple[GammaTensor, GammaTensor]:
        return self // other, self % other

    def divmod(self, other: Any) -> Tuple[GammaTensor, GammaTensor]:
        return self.__divmod__(other)

    def __matmul__(self, other: Any) -> GammaTensor:
        # relative
        from .phi_tensor import PhiTensor

        output_state = dict()
        # Add this tensor to the chain
        output_state[self.id] = self

        if isinstance(other, PhiTensor):
            other = other.gamma

        if isinstance(other, GammaTensor):
            output_state[other.id] = other
            child = self.child @ other.child
            min_min = (self.min_vals @ other.min_vals).data
            min_max = (self.min_vals @ other.max_vals).data
            max_max = (self.max_vals @ other.max_vals).data
            max_min = (self.max_vals @ other.min_vals).data
            minv = np.min([min_min, min_max, max_max, max_min], axis=0)  # type: ignore
            min_val = lazyrepeatarray(data=minv, shape=child.shape)
            max_val = self.max_vals.__matmul__(other.max_vals)
            output_ds = self.data_subjects @ other.data_subjects

        else:
            child = self.child @ other
            min_val = self.min_vals.__matmul__(other)
            max_val = self.max_vals.__matmul__(other)

            output_ds = self.data_subjects @ other
            output_state[np.random.randint(low=0, high=2**31 - 1)] = other

        return GammaTensor(
            child=child,
            data_subjects=output_ds,
            min_vals=min_val,
            max_vals=max_val,
            func_str=GAMMA_TENSOR_OP.MATMUL.value,
            sources=output_state,
        )

    def searchsorted(self, v: Any) -> GammaTensor:
        raise NotImplementedError

    def __rmatmul__(self, other: Any) -> GammaTensor:
        # relative
        from .phi_tensor import PhiTensor

        output_state = dict()
        # Add this tensor to the chain
        output_state[self.id] = self

        if isinstance(other, PhiTensor):
            other = other.gamma

        if isinstance(other, GammaTensor):
            output_state[other.id] = other
            child = self.child.__rmatmul__(other.child)
            min_val = self.min_vals.__rmatmul__(other.min_vals)
            max_val = self.max_vals.__rmatmul__(other.max_vals)
            output_ds = self.data_subjects.__rmatmul__(other.data_subjects)

        else:
            child = self.child.__rmatmul__(other)
            min_val = self.min_vals.__rmatmul__(other)
            max_val = self.max_vals.__rmatmul__(other)
            output_ds = self.data_subjects.__rmatmul__(other)
            output_state[np.random.randint(low=0, high=2**31 - 1)] = other

        return GammaTensor(
            child=child,
            data_subjects=output_ds,
            min_vals=min_val,
            max_vals=max_val,
            func_str=GAMMA_TENSOR_OP.RMATMUL.value,
            sources=output_state,
        )

    def __gt__(self, other: Any) -> GammaTensor:
        # relative
        from .phi_tensor import PhiTensor

        output_state = dict()
        # Add this tensor to the chain
        output_state[self.id] = self

        if isinstance(other, PhiTensor):
            other = other.gamma

        if isinstance(other, GammaTensor):
            output_state[other.id] = other
            child = self.child.__gt__(other.child)
            output_ds = self.data_subjects + other.data_subjects

        else:
            output_state[np.random.randint(low=0, high=2**31 - 1)] = other
            child = self.child.__gt__(other)
            output_ds = self.data_subjects

        min_val = self.min_vals * 0
        max_val = (self.max_vals * 0) + 1

        return GammaTensor(
            child=child,
            data_subjects=output_ds,
            min_vals=min_val,
            max_vals=max_val,
            func_str=GAMMA_TENSOR_OP.GREATER.value,
            sources=output_state,
        )

    def __ge__(self, other: Any) -> GammaTensor:
        # relative
        from .phi_tensor import PhiTensor

        output_state = dict()
        # Add this tensor to the chain
        output_state[self.id] = self

        if isinstance(other, PhiTensor):
            other = other.gamma

        if isinstance(other, GammaTensor):
            output_state[other.id] = other
            child = self.child.__ge__(other.child)
            output_ds = self.data_subjects + other.data_subjects

        else:
            output_state[np.random.randint(low=0, high=2**31 - 1)] = other
            child = self.child.__ge__(other)
            output_ds = self.data_subjects

        min_val = self.min_vals * 0
        max_val = (self.max_vals * 0) + 1

        return GammaTensor(
            child=child,
            data_subjects=output_ds,
            min_vals=min_val,
            max_vals=max_val,
            func_str=GAMMA_TENSOR_OP.GREATER_EQUAL.value,
            sources=output_state,
        )

    def __eq__(self, other: Any) -> GammaTensor:  # type: ignore
        # relative
        from .phi_tensor import PhiTensor

        output_state = dict()
        # Add this tensor to the chain
        output_state[self.id] = self

        if isinstance(other, PhiTensor):
            other = other.gamma

        if isinstance(other, GammaTensor):
            output_state[other.id] = other
            child = self.child.__eq__(other.child)
            output_ds = self.data_subjects + other.data_subjects

        else:
            output_state[np.random.randint(low=0, high=2**31 - 1)] = other
            child = self.child.__eq__(other)
            output_ds = self.data_subjects

        min_val = self.min_vals * 0
        max_val = (self.max_vals * 0) + 1

        return GammaTensor(
            child=child,
            data_subjects=output_ds,
            min_vals=min_val,
            max_vals=max_val,
            func_str=GAMMA_TENSOR_OP.EQUAL.value,
            sources=output_state,
        )

    def __ne__(self, other: Any) -> GammaTensor:  # type: ignore
        # relative
        from .phi_tensor import PhiTensor

        output_state = dict()
        # Add this tensor to the chain
        output_state[self.id] = self

        if isinstance(other, PhiTensor):
            other = other.gamma

        if isinstance(other, GammaTensor):
            output_state[other.id] = other
            child = self.child.__ne__(other.child)
            output_ds = self.data_subjects + other.data_subjects

        else:
            output_state[np.random.randint(low=0, high=2**31 - 1)] = other
            child = self.child.__ne__(other)
            output_ds = self.data_subjects

        min_val = self.min_vals * 0
        max_val = (self.max_vals * 0) + 1

        return GammaTensor(
            child=child,
            data_subjects=output_ds,
            min_vals=min_val,
            max_vals=max_val,
            func_str=GAMMA_TENSOR_OP.NOT_EQUAL.value,
            sources=output_state,
        )

    def __lt__(self, other: Any) -> GammaTensor:
        # relative
        from .phi_tensor import PhiTensor

        output_state = dict()
        # Add this tensor to the chain
        output_state[self.id] = self

        if isinstance(other, PhiTensor):
            other = other.gamma

        if isinstance(other, GammaTensor):
            output_state[other.id] = other
            child = self.child.__lt__(other.child)
            output_ds = self.data_subjects + other.data_subjects

        else:
            output_state[np.random.randint(low=0, high=2**31 - 1)] = other
            child = self.child.__lt__(other)
            output_ds = self.data_subjects

        min_val = self.min_vals * 0
        max_val = (self.max_vals * 0) + 1

        return GammaTensor(
            child=child,
            data_subjects=output_ds,
            min_vals=min_val,
            max_vals=max_val,
            func_str=GAMMA_TENSOR_OP.LESS.value,
            sources=output_state,
        )

    def __le__(self, other: Any) -> GammaTensor:
        # relative
        from .phi_tensor import PhiTensor

        output_state = dict()
        # Add this tensor to the chain
        output_state[self.id] = self

        if isinstance(other, PhiTensor):
            other = other.gamma

        if isinstance(other, GammaTensor):
            output_state[other.id] = other
            child = self.child.__le__(other.child)
            output_ds = self.data_subjects + other.data_subjects

        else:
            output_state[np.random.randint(low=0, high=2**31 - 1)] = other
            child = self.child.__le__(other)
            output_ds = self.data_subjects

        min_val = self.min_vals * 0
        max_val = (self.max_vals * 0) + 1

        return GammaTensor(
            child=child,
            data_subjects=output_ds,
            min_vals=min_val,
            max_vals=max_val,
            func_str=GAMMA_TENSOR_OP.LESS_EQUAL.value,
            sources=output_state,
        )

    def __abs__(self) -> GammaTensor:

        output_state = dict()
        output_state[self.id] = self

        child = self.child.__abs__()

        min_val = abs(self.min_vals.data)
        max_val = abs(self.max_vals.data)

        new_min_val = min(min_val, max_val)
        new_max_val = max(min_val, max_val)

        return GammaTensor(
            child=child,
            data_subjects=self.data_subjects,
            min_vals=lazyrepeatarray(data=new_min_val, shape=child.shape),
            max_vals=lazyrepeatarray(data=new_max_val, shape=child.shape),
            func_str=GAMMA_TENSOR_OP.ABS.value,
            sources=output_state,
        )

    def argmax(
        self,
        axis: Optional[int] = None,
    ) -> GammaTensor:

        output_state = dict()
        output_state[self.id] = self

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

        return GammaTensor(
            child=child,
            data_subjects=data_subjects,
            min_vals=lazyrepeatarray(data=0, shape=child.shape),
            max_vals=lazyrepeatarray(data=max_value, shape=child.shape),
            func_str=GAMMA_TENSOR_OP.ARGMAX.value,
            sources=output_state,
        )

    def argmin(
        self,
        axis: Optional[int] = None,
    ) -> GammaTensor:

        output_state = dict()
        output_state[self.id] = self

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

        return GammaTensor(
            child=child,
            data_subjects=data_subjects,
            min_vals=lazyrepeatarray(data=0, shape=child.shape),
            max_vals=lazyrepeatarray(data=max_value, shape=child.shape),
            func_str=GAMMA_TENSOR_OP.ARGMIN.value,
            sources=output_state,
        )

    def log(self) -> GammaTensor:
        output_state = dict()
        output_state[self.id] = self

        if isinstance(self.min_vals, lazyrepeatarray):
            min_val = lazyrepeatarray(
                data=np.log(self.min_vals.data.min()), shape=self.shape
            )
            max_val = lazyrepeatarray(
                data=np.log(self.max_vals.data.max()), shape=self.shape
            )
        elif isinstance(self.min_vals, np.ndarray):
            min_val = lazyrepeatarray(data=np.log(self.min_vals), shape=self.shape)
            max_val = lazyrepeatarray(data=np.log(self.max_vals), shape=self.shape)
        elif isinstance(self.min_vals, (int, float)):
            min_val = lazyrepeatarray(data=np.log(self.min_vals), shape=self.shape)
            max_val = lazyrepeatarray(data=np.log(self.max_vals), shape=self.shape)
        else:
            raise NotImplementedError(
                f"Undefined behaviour for type: {type(self.min_vals)}"
            )

        return GammaTensor(
            child=np.log(self.child),
            min_vals=min_val,
            max_vals=max_val,
            data_subjects=self.data_subjects,
            func_str=GAMMA_TENSOR_OP.LOG.value,
            sources=output_state,
        )

    def flatten(self, order: str = "C") -> GammaTensor:
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
            GammaTensor
        A copy of the input array, flattened to one dimension.

        """

        if order == "C":
            func = GAMMA_TENSOR_OP.FLATTEN_C.value
        elif order == "F":
            func = GAMMA_TENSOR_OP.FLATTEN_F.value
        elif order == "A":
            func = GAMMA_TENSOR_OP.FLATTEN_A.value
        elif order == "K":
            func = GAMMA_TENSOR_OP.FLATTEN_K.value
        else:
            raise NotImplementedError(f"Flatten not implemented for order={order}")

        output_sources = dict()
        output_sources[self.id] = self

        result = self.child.flatten(order)
        return GammaTensor(
            child=result,
            data_subjects=self.data_subjects.flatten(order),
            min_vals=lazyrepeatarray(data=self.min_vals.data, shape=result.shape),
            max_vals=lazyrepeatarray(data=self.max_vals.data, shape=result.shape),
            is_linear=True,
            func_str=func,
            sources=output_sources,
        )

    def transpose(self, *args: Any, **kwargs: Any) -> GammaTensor:
        output_state = dict()
        # Add this tensor to the chain
        output_state[self.id] = self

        output_ds = self.data_subjects.transpose(*args, **kwargs)
        output_data = self.child.transpose(*args, **kwargs)

        min_vals = lazyrepeatarray(data=self.min_vals.data, shape=output_data.shape)
        max_vals = lazyrepeatarray(data=self.max_vals.data, shape=output_data.shape)

        return GammaTensor(
            child=output_data,
            data_subjects=output_ds,
            min_vals=min_vals,
            max_vals=max_vals,
            func_str=GAMMA_TENSOR_OP.TRANSPOSE.value,
            sources=output_state,
        )

    @property
    def T(self) -> GammaTensor:
        return self.transpose()

    def sum(
        self,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: Optional[bool] = False,
        initial: Optional[float] = None,
        where: Optional[ArrayLike] = None,
    ) -> GammaTensor:
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
        sources = dict()
        sources[self.id] = self
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

        if not isinstance(result, np.ndarray):
            result = np.array(result)

        return GammaTensor(
            child=result,
            data_subjects=np.array(output_ds),
            min_vals=lazyrepeatarray(data=self.min_vals.data * num, shape=result.shape),
            max_vals=lazyrepeatarray(data=self.max_vals.data * num, shape=result.shape),
            func_str=GAMMA_TENSOR_OP.SUM.value,
            sources=sources,
        )

    def __pow__(
        self, power: Union[float, int], modulo: Optional[int] = None
    ) -> GammaTensor:
        sources = dict()
        sources[self.id] = self
        sources["0"] = power  # type: ignore
        if modulo is None:
            if self.min_vals.data <= 0 <= self.max_vals.data:
                # If data is in range [-5, 5], it's possible the minimum is 0 and not (-5)^2
                minv = min(0, (self.min_vals.data**power).min())
            else:
                minv = self.min_vals.data**power

            return GammaTensor(
                child=self.child**power,
                data_subjects=self.data_subjects,
                min_vals=lazyrepeatarray(data=minv, shape=self.shape),
                max_vals=lazyrepeatarray(
                    data=self.max_vals.data**power, shape=self.shape
                ),
                func_str=GAMMA_TENSOR_OP.POWER.value,
                sources=sources,
            )
        else:
            # This may be unnecessary- modulo is NotImplemented in ndarray.pow
            if self.min_vals.data <= 0 <= self.max_vals.data:
                # If data is in range [-5, 5], it's possible the minimum is 0 and not (-5)^2
                minv = min(0, (self.min_vals.data**power).min() % modulo)
            else:
                minv = (self.min_vals.data**power) % modulo

            return GammaTensor(
                child=(self.child**power) % modulo,
                data_subjects=self.data_subjects,
                min_vals=lazyrepeatarray(data=minv, shape=self.shape),
                max_vals=lazyrepeatarray(
                    data=(self.max_vals.data**power) % modulo, shape=self.shape
                ),
                func_str=GAMMA_TENSOR_OP.POWER.value,
                sources=sources,
            )

    def ones_like(self, *args: Any, **kwargs: Any) -> GammaTensor:
        output_state = dict()
        output_state[self.id] = self

        child = (
            np.ones_like(self.child, *args, **kwargs)
            if isinstance(self.child, np.ndarray)
            else self.child.ones_like(*args, **kwargs)
        )

        min_val = self.min_vals.ones_like(*args, **kwargs)
        max_val = self.max_vals.ones_like(*args, **kwargs)

        return GammaTensor(
            child=child,
            data_subjects=self.data_subjects,
            min_vals=min_val,
            max_vals=max_val,
            func_str=GAMMA_TENSOR_OP.ONES_LIKE.value,
            sources=output_state,
        )

    def zeros_like(self, *args: Any, **kwargs: Any) -> GammaTensor:
        output_state = dict()
        output_state[self.id] = self

        child = (
            np.zeros_like(self.child, *args, **kwargs)
            if not hasattr(self.child, "zeros_like")
            else self.child.zeros_like(*args, **kwargs)
        )

        min_val = self.min_vals.zeros_like(*args, **kwargs)
        max_val = self.max_vals.zeros_like(*args, **kwargs)

        return GammaTensor(
            child=child,
            data_subjects=self.data_subjects,
            min_vals=min_val,
            max_vals=max_val,
            func_str=GAMMA_TENSOR_OP.ZEROS_LIKE.value,
            sources=output_state,
        )

    def filtered(self) -> GammaTensor:
        # This is only used during publish to filter out data in GammaTensors with no_op. It serves no other purpose.
        return GammaTensor(
            child=jnp.zeros_like(self.child),
            data_subjects=self.data_subjects,
            min_vals=self.min_vals * 0,
            max_vals=self.max_vals * 1,
            func_str=GAMMA_TENSOR_OP.NOOP.value,
        )

    def ravel(self, order: Optional[str] = "C") -> GammaTensor:
        output_state = dict()
        output_state[self.id] = self

        data = self.child
        output_data = data.ravel(order=order)

        output_data_subjects = self.data_subjects.ravel(order=order)

        min_val = lazyrepeatarray(data=self.min_vals.data, shape=output_data.shape)
        max_val = lazyrepeatarray(data=self.max_vals.data, shape=output_data.shape)

        return GammaTensor(
            child=output_data,
            data_subjects=output_data_subjects,
            min_vals=min_val,
            max_vals=max_val,
            func_str=GAMMA_TENSOR_OP.RAVEL.value,
            sources=output_state,
        )

    def resize(
        self, new_shape: Union[int, Tuple[int, ...]], refcheck: bool = True
    ) -> GammaTensor:
        output_state = dict()
        output_state[self.id] = self

        self.child.resize(new_shape, refcheck=refcheck)
        self.data_subjects.resize(new_shape, refcheck=refcheck)

        min_val = lazyrepeatarray(data=self.min_vals.data, shape=self.child.shape)
        max_val = lazyrepeatarray(data=self.max_vals.data, shape=self.child.shape)

        return GammaTensor(
            child=self.child,
            data_subjects=self.data_subjects,
            min_vals=min_val,
            max_vals=max_val,
            func_str=GAMMA_TENSOR_OP.RESIZE.value,
            sources=output_state,
        )

    def compress(
        self, condition: List[bool], axis: Optional[int] = None
    ) -> GammaTensor:
        output_state = dict()
        output_state[self.id] = self

        data = self.child
        output_data = data.compress(condition, axis)
        if 0 in output_data.shape:
            raise NotImplementedError
        output_data_subjects = self.data_subjects.compress(condition, axis)

        min_val = lazyrepeatarray(data=self.min_vals.data, shape=output_data.shape)
        max_val = lazyrepeatarray(data=self.max_vals.data, shape=output_data.shape)

        return GammaTensor(
            child=output_data,
            data_subjects=output_data_subjects,
            min_vals=min_val,
            max_vals=max_val,
            func_str=GAMMA_TENSOR_OP.COMPRESS.value,
            sources=output_state,
        )

    def squeeze(
        self, axis: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> GammaTensor:
        output_state = dict()
        output_state[self.id] = self

        data = self.child
        output_data = np.squeeze(data, axis)
        output_data_subjects = np.squeeze(self.data_subjects, axis)

        min_val = lazyrepeatarray(data=self.min_vals.data, shape=output_data.shape)
        max_val = lazyrepeatarray(data=self.max_vals.data, shape=output_data.shape)

        return GammaTensor(
            child=output_data,
            data_subjects=output_data_subjects,
            min_vals=min_val,
            max_vals=max_val,
            func_str=GAMMA_TENSOR_OP.SQUEEZE.value,
            sources=output_state,
        )

    def any(
        self,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: Optional[bool] = False,
        where: Optional[ArrayLike] = None,
    ) -> GammaTensor:
        output_state = dict()
        output_state[self.id] = self

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
                initial=DataSubjectArray(),
                where=where,
            )

        return GammaTensor(
            child=out_child,
            data_subjects=new_data_subjects,
            min_vals=lazyrepeatarray(data=0, shape=out_child.shape),
            max_vals=lazyrepeatarray(data=1, shape=out_child.shape),
            func_str=GAMMA_TENSOR_OP.ANY.value,
            sources=output_state,
        )

    def all(
        self,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: Optional[bool] = False,
        where: Optional[ArrayLike] = None,
    ) -> GammaTensor:
        output_state = dict()
        output_state[self.id] = self

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
                initial=DataSubjectArray(),
                where=where,
            )

        return GammaTensor(
            child=out_child,
            data_subjects=new_data_subjects,
            min_vals=lazyrepeatarray(data=0, shape=out_child.shape),
            max_vals=lazyrepeatarray(data=1, shape=out_child.shape),
            func_str=GAMMA_TENSOR_OP.ALL.value,
            sources=output_state,
        )

    def __and__(self, other: Any) -> GammaTensor:
        # relative
        from .phi_tensor import PhiTensor

        output_state = dict()
        # Add this tensor to the chain
        output_state[self.id] = self

        if isinstance(other, PhiTensor):
            other = other.gamma

        if isinstance(other, GammaTensor):
            output_state[other.id] = other

            child = self.child & other.child
            output_ds = self.data_subjects + other.data_subjects
            other_min, other_max = other.min_vals.data, other.max_vals.data

        elif is_acceptable_simple_type(other):
            output_state[np.random.randint(low=0, high=2**31 - 1)] = other

            child = self.child & other
            output_ds = self.data_subjects
            if isinstance(other, np.ndarray):
                other_min, other_max = other.min(), other.max()
            else:
                other_min, other_max = other, other
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

        return GammaTensor(
            child=child,
            data_subjects=output_ds,
            min_vals=lazyrepeatarray(data=_min_vals, shape=child.shape),
            max_vals=lazyrepeatarray(data=_max_vals, shape=child.shape),
            func_str=GAMMA_TENSOR_OP.LOGICAL_AND.value,
            sources=output_state,
        )

    def __or__(self, other: Any) -> GammaTensor:
        # relative
        from .phi_tensor import PhiTensor

        output_state = dict()
        # Add this tensor to the chain
        output_state[self.id] = self

        if isinstance(other, PhiTensor):
            other = other.gamma

        if isinstance(other, GammaTensor):
            output_state[other.id] = other

            child = self.child | other.child
            output_ds = self.data_subjects + other.data_subjects
            other_min, other_max = other.min_vals.data, other.max_vals.data

        elif is_acceptable_simple_type(other):
            if isinstance(other, np.ndarray):
                other_min, other_max = other.min(), other.max()
            else:
                other_min, other_max = other, other
            output_state[np.random.randint(low=0, high=2**31 - 1)] = other

            child = self.child | other
            output_ds = self.data_subjects
        else:
            print("Type is unsupported:" + str(type(other)))
            raise NotImplementedError

        # TODO: should modify for a tighter found for or
        _max = int(max(self.max_vals.data, other_max))
        _min_vals = min(self.min_vals.data, other_min)
        _max_vals = (2 ** (_max).bit_length()) - 1

        return GammaTensor(
            child=child,
            data_subjects=output_ds,
            min_vals=lazyrepeatarray(data=_min_vals, shape=child.shape),
            max_vals=lazyrepeatarray(data=_max_vals, shape=child.shape),
            func_str=GAMMA_TENSOR_OP.LOGICAL_OR.value,
            sources=output_state,
        )

    def __pos__(self) -> GammaTensor:
        output_state = dict()
        output_state[self.id] = self

        return GammaTensor(
            child=self.child,
            data_subjects=self.data_subjects,
            min_vals=self.min_vals,
            max_vals=self.max_vals,
            func_str=GAMMA_TENSOR_OP.POSITIVE.value,
            sources=output_state,
        )

    def __neg__(self) -> GammaTensor:
        output_state = dict()
        output_state[self.id] = self

        return GammaTensor(
            child=self.child * -1,
            data_subjects=self.data_subjects,
            min_vals=self.max_vals * -1,
            max_vals=self.min_vals * -1,
            func_str=GAMMA_TENSOR_OP.NEGATIVE.value,
            sources=output_state,
        )

    def reshape(self, shape: Tuple[int, ...]) -> GammaTensor:
        sources = dict()
        sources[self.id] = self
        sources["0"] = shape  # type: ignore
        output_data = self.child.reshape(shape)
        return GammaTensor(
            child=output_data,
            data_subjects=self.data_subjects.reshape(shape),
            min_vals=lazyrepeatarray(data=self.min_vals.data, shape=output_data.shape),
            max_vals=lazyrepeatarray(data=self.max_vals.data, shape=output_data.shape),
            func_str=GAMMA_TENSOR_OP.RESHAPE.value,
            sources=sources,
        )

    def _argmax(self, axis: Optional[int]) -> np.ndarray:
        raise NotImplementedError
        # return self.child.argmax(axis)

    def mean(self, axis: Union[int, Tuple[int, ...]], **kwargs: Any) -> GammaTensor:
        output_state = dict()
        output_state[self.id] = self

        result = self.child.mean(axis, **kwargs)

        return GammaTensor(
            child=result,
            data_subjects=self.data_subjects.mean(axis, **kwargs),
            min_vals=lazyrepeatarray(data=self.min_vals.data, shape=result.shape),
            max_vals=lazyrepeatarray(data=self.max_vals.data, shape=result.shape),
            sources=output_state,
            func_str=GAMMA_TENSOR_OP.MEAN.value,
        )

    def expand_dims(self, axis: Optional[int] = None) -> GammaTensor:
        raise NotImplementedError
        # result = np.expand_dims(self.child, axis)

        # target_shape_dsl = list(self.data_subjects.shape)
        # if axis:
        #     target_shape_dsl.insert(axis + 1, 1)

        # return GammaTensor(
        #     child=result,
        #     data_subjects=np.expand_dims(self.data_subjects, axis),
        #     min_vals=lazyrepeatarray(data=self.min_vals.data, shape=result.shape),
        #     max_vals=lazyrepeatarray(data=self.max_vals.data, shape=result.shape),
        # )

    def std(
        self, axis: Optional[Union[int, Tuple[int, ...]]] = None, **kwargs: Any
    ) -> GammaTensor:
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

            standard_deviation: GammaTensor
        """
        output_state = dict()
        output_state[self.id] = self

        result = self.child.std(axis, **kwargs)
        minv = (
            self.min_vals.data
            if isinstance(self.min_vals, lazyrepeatarray)
            else self.min_vals
        )
        maxv = (
            self.max_vals.data
            if isinstance(self.max_vals, lazyrepeatarray)
            else self.max_vals
        )
        return GammaTensor(
            child=result,
            data_subjects=self.data_subjects.std(axis, **kwargs),
            min_vals=lazyrepeatarray(data=0, shape=result.shape),
            max_vals=lazyrepeatarray(data=(maxv - minv) / 2, shape=result.shape),
            sources=output_state,
            func_str=GAMMA_TENSOR_OP.STD.value,
        )

    def var(
        self, axis: Optional[Union[int, Tuple[int, ...]]] = None, **kwargs: Any
    ) -> GammaTensor:
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

        output_state = dict()
        output_state[self.id] = self

        result = self.child.var(axis, **kwargs)
        minv = (
            self.min_vals.data
            if isinstance(self.min_vals, lazyrepeatarray)
            else self.min_vals
        )
        maxv = (
            self.max_vals.data
            if isinstance(self.max_vals, lazyrepeatarray)
            else self.max_vals
        )
        return GammaTensor(
            child=result,
            data_subjects=self.data_subjects.var(axis, **kwargs),
            min_vals=lazyrepeatarray(data=0, shape=result.shape),
            max_vals=lazyrepeatarray(
                data=0.25 * (maxv - minv) ** 2, shape=result.shape
            ),
            sources=output_state,
            func_str=GAMMA_TENSOR_OP.VAR.value,
        )

    def dot(self, other: Union[np.ndarray, GammaTensor]) -> GammaTensor:
        # TODO: These bounds might not be super tight- if min,max = [-1, 1], there might be a dot product
        # such that the minimum value should be 0
        if isinstance(other, np.ndarray):
            raise NotImplementedError
            # result = jnp.dot(self.child, other)

            # output_ds = self.data_subjects.dot(other)

            # if isinstance(self.min_vals, lazyrepeatarray):
            #     minv = lazyrepeatarray(
            #         data=jnp.dot(
            #             np.ones_like(self.child) * self.min_vals.data, other
            #         ).min(),
            #         shape=result.shape,
            #     )
            #     maxv = lazyrepeatarray(
            #         data=jnp.dot(
            #             np.ones_like(self.child) * self.max_vals.data, other
            #         ).max(),
            #         shape=result.shape,
            #     )

            # elif isinstance(self.min_vals, (int, float)):
            #     minv = lazyrepeatarray(
            #         data=jnp.dot(np.ones_like(self.child) * self.min_vals, other).min(),
            #         shape=result.shape,
            #     )
            #     maxv = lazyrepeatarray(
            #         data=jnp.dot(np.ones_like(self.child) * self.max_vals, other).max(),
            #         shape=result.shape,
            #     )
            # else:
            #     raise NotImplementedError

            # return GammaTensor(
            #     child=result,
            #     data_subjects=output_ds,
            #     min_vals=minv,
            #     max_vals=maxv,
            # )
        elif isinstance(other, GammaTensor):
            output_state = dict()
            output_state[self.id] = self
            output_state[other.id] = other

            output_ds = self.data_subjects.dot(other.data_subjects)

            result = jnp.dot(self.child, other.child)

            if isinstance(self.min_vals, lazyrepeatarray):

                minv = lazyrepeatarray(
                    data=jnp.dot(
                        np.ones_like(self.child) * self.min_vals.data,
                        np.ones_like(other.child) * other.min_vals.data,
                    ).min(),
                    shape=result.shape,
                )
                maxv = lazyrepeatarray(
                    data=jnp.dot(
                        np.ones_like(self.child) * self.max_vals.data,
                        np.ones_like(other.child) * other.max_vals.data,
                    ).max(),
                    shape=result.shape,
                )
            elif isinstance(self.min_vals, (int, float)):
                minv = lazyrepeatarray(
                    data=jnp.dot(
                        np.ones_like(self.child) * self.min_vals,
                        np.ones_like(other.child) * other.min_vals,
                    ).min(),
                    shape=result.shape,
                )
                maxv = lazyrepeatarray(
                    data=jnp.dot(
                        np.ones_like(self.child) * self.max_vals,
                        np.ones_like(other.child) * other.max_vals,
                    ).max(),
                    shape=result.shape,
                )
            else:
                raise NotImplementedError

            return GammaTensor(
                child=result,
                data_subjects=output_ds,
                min_vals=minv,
                max_vals=maxv,
                func_str=GAMMA_TENSOR_OP.DOT.value,
                sources=output_state,
            )
        else:
            raise NotImplementedError(
                f"Undefined behaviour for GT.dot with {type(other)}"
            )

    def sqrt(self) -> GammaTensor:
        state = dict()
        state.update(self.sources)

        min_v = jnp.sqrt(self.min_vals.data)
        max_v = jnp.sqrt(self.min_vals.data)

        child = jnp.sqrt(self.child)
        min_val = lazyrepeatarray(min_v, shape=child.shape)
        max_val = lazyrepeatarray(max_v, shape=child.shape)

        return GammaTensor(
            child=child,
            data_subjects=self.data_subjects,
            min_vals=min_val,
            max_vals=max_val,
            func_str=GAMMA_TENSOR_OP.SQRT.value,
            sources=state,
        )

    def abs(self) -> GammaTensor:
        state = dict()
        state.update(self.sources)

        data = self.child
        output = np.abs(data)

        min_v = np.abs(self.min_vals.data)
        max_v = np.abs(self.min_vals.data)

        return GammaTensor(
            child=output,
            data_subjects=self.data_subjects,
            min_vals=lazyrepeatarray(min_v, shape=output.shape),
            max_vals=lazyrepeatarray(max_v, shape=output.shape),
            func_str=GAMMA_TENSOR_OP.ABS.value,
            sources=state,
        )

    def clip(self, a_min: float, a_max: float) -> GammaTensor:
        state = dict()
        state.update(self.sources)

        output_data = self.child.clip(a_min, a_max)

        min_v = np.clip(self.min_vals.data, a_min, a_max)
        max_v = np.clip(self.max_vals.data, a_min, a_max)

        min_vals = lazyrepeatarray(data=min_v, shape=output_data.shape)
        max_vals = lazyrepeatarray(data=max_v, shape=output_data.shape)

        return GammaTensor(
            child=output_data,
            data_subjects=self.data_subjects,
            min_vals=min_vals,
            max_vals=max_vals,
            func_str=GAMMA_TENSOR_OP.CLIP.value,
            sources=state,
        )

    def nonzero(self) -> GammaTensor:
        output_state = dict()
        output_state[self.id] = self

        out_child = np.array(np.nonzero(self.child))
        no_axis = len(self.child.shape)
        out_data_subjects = np.repeat(
            np.array([self.data_subjects[self.child != 0]]), no_axis, axis=0
        )

        min_vals = lazyrepeatarray(data=0, shape=out_child.shape)
        max_vals = lazyrepeatarray(data=max(self.child.shape), shape=out_child.shape)

        return GammaTensor(
            child=out_child,
            data_subjects=out_data_subjects,
            min_vals=min_vals,
            max_vals=max_vals,
            func_str=GAMMA_TENSOR_OP.NONZERO.value,
            sources=output_state,
        )

    def swapaxes(self, axis1: int, axis2: int) -> GammaTensor:
        output_state = dict()
        output_state[self.id] = self

        out_child = np.swapaxes(self.child, axis1, axis2)
        data_subjects = np.swapaxes(self.data_subjects, axis1, axis2)

        min_vals = lazyrepeatarray(data=self.min_vals.data, shape=out_child.shape)
        max_vals = lazyrepeatarray(data=self.max_vals.data, shape=out_child.shape)

        return GammaTensor(
            child=out_child,
            data_subjects=data_subjects,
            min_vals=min_vals,
            max_vals=max_vals,
            func_str=GAMMA_TENSOR_OP.SWAPAXES.value,
            sources=output_state,
        )

    @staticmethod
    def convert_dsl(state: dict, new_state: Optional[dict] = None) -> Dict:
        if new_state is None:
            new_state = dict()
        if state:
            for tensor in list(state.values()):
                if isinstance(tensor.data_subjects, np.ndarray):
                    new_tensor = GammaTensor(
                        child=tensor.child,
                        data_subjects=np.zeros_like(
                            tensor.data_subjects, dtype=np.int64
                        ),
                        min_vals=tensor.min_vals,
                        max_vals=tensor.max_vals,
                        func_str=tensor.func_str,
                        sources=GammaTensor.convert_dsl(tensor.sources, {}),
                    )
                    # for idx, row in enumerate(tensor.data_subjects):
                    #     tensor.data_subjects[idx] = jnp.zeros_like(np.zeros_like(row), jnp.int64)
                else:

                    new_tensor = tensor
                new_state[new_tensor.id] = new_tensor
            return new_state
        else:
            return {}

    def publish(
        self,
        get_budget_for_user: Callable,
        deduct_epsilon_for_user: Callable,
        ledger: DataSubjectLedger,
        sigma: float,
        private: bool,
    ) -> np.ndarray:

        if (
            not self.sources
        ):  # if state tree is empty (e.g. publishing a PhiTensor w/ public vals directly)
            self.sources[self.id] = self

        return publish(
            tensor=self,
            ledger=ledger,
            get_budget_for_user=get_budget_for_user,
            deduct_epsilon_for_user=deduct_epsilon_for_user,
            sigma=sigma,
            is_linear=self.is_linear,
            private=private,
        )

    # def expand_dims(self, axis: int) -> GammaTensor:
    #     def _expand_dims(state: dict) -> jax.numpy.DeviceArray:
    #         return jnp.expand_dims(self.run(state), axis)
    #
    #     state = dict()
    #     state.update(self.state)
    #
    #     return GammaTensor(
    #         child=jnp.expand_dims(self.child, axis),
    #         data_subjects=self.data_subjects,
    #         min_vals=self.min_vals,
    #         max_vals=self.max_vals,
    #         func=_expand_dims,
    #         sources=state,
    #     )

    def __len__(self) -> int:
        if not hasattr(self.child, "__len__"):
            if self.child is None:
                return 0
            return 1
        try:
            return len(self.child)
        except Exception:  # nosec
            return self.child.size

    def __getitem__(self, item: Union[int, slice, PassthroughTensor]) -> GammaTensor:
        # TODO: Technically we could reduce ds.one_hot_lookup to remove any DS that won't be there
        # There technically isn't any penalty for keeping it as is, but maybe there's a sidechannel attack
        # where you index into one value in a GammaTensor and get all the data subjects of that Tensor?

        if isinstance(self.min_vals, (int, float)):
            minv = self.min_vals
            maxv = self.max_vals
        elif isinstance(self.min_vals, lazyrepeatarray):
            minv = self.min_vals[item]  # type: ignore
            maxv = self.max_vals[item]
        else:
            raise NotImplementedError

        if isinstance(item, PassthroughTensor):
            data = self.child[item.child]
            if self.shape == self.data_subjects.shape:
                return GammaTensor(
                    child=data,
                    min_vals=minv,
                    max_vals=maxv,
                    data_subjects=self.data_subjects[item.child],
                )
            elif len(self.shape) < len(self.data_subjects.shape):
                return GammaTensor(
                    child=data,
                    min_vals=minv,
                    max_vals=maxv,
                    data_subjects=self.data_subjects[item.child],
                    # self.data_subjects.data_subjects_indexed[:, item.child],
                )
            else:
                raise Exception(
                    f"Incompatible shapes: {self.shape}, {self.data_subjects.shape}"
                )
        else:
            data = self.child[item]

            return GammaTensor(
                child=data,
                min_vals=minv,
                max_vals=maxv,
                data_subjects=self.data_subjects[item],
            )

    def __setitem__(
        self, key: Union[int, slice, NDArray], value: Union[GammaTensor, np.ndarray]
    ) -> None:
        # relative
        from .phi_tensor import PhiTensor

        if isinstance(value, (PhiTensor, GammaTensor)):
            self.child[key] = value.child
            minv = value.child.min()
            maxv = value.child.max()

            if minv < self.min_vals.data.min():
                self.min_vals.data = minv

            if maxv > self.max_vals.data.max():
                self.max_vals.data = maxv

            self.data_subjects[key] = value.data_subjects

            # output_dsl = DataSubjectList.insert(
            #     dsl1=self.data_subjects, dsl2=value.data_subjects, index=key
            # )
            # self.data_subjects.one_hot_lookup = output_dsl.one_hot_lookup
            # self.data_subjects.data_subjects_indexed = output_dsl.data_subjects_indexed

        elif isinstance(value, np.ndarray):
            self.child[key] = value
            minv = value.min()
            maxv = value.max()

            if minv < self.min_vals.data.min():
                self.min_vals.data = minv

            if maxv > self.max_vals.data.max():
                self.max_vals.data = maxv

        else:
            raise NotImplementedError

    def copy(self, order: str = "C") -> GammaTensor:
        """
        Return a copy of the array.

        Parameters
            order:  {‘C’, ‘F’, ‘A’, ‘K’}, optional

        Controls the memory layout of the copy.
        ‘C’ means C-order, ‘F’ means F-order,
        ‘A’ means ‘F’ if a is Fortran contiguous,
        ‘C’ otherwise.
        ‘K’ means match the layout of a as closely as possible.
        (Note that this function and numpy.copy are very similar but have different default values
        for their order= arguments, and this function always passes sub-classes through.)
        """
        output_state = dict()
        output_state[self.id] = self

        return GammaTensor(
            child=self.child.copy(order),
            data_subjects=self.data_subjects.copy(order),
            min_vals=self.min_vals.copy(order),
            max_vals=self.max_vals.copy(order),
            func_str=GAMMA_TENSOR_OP.COPY.value,
            sources=output_state,
        )

    def ptp(
        self,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
    ) -> GammaTensor:
        output_state = dict()
        output_state[self.id] = self

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

        return GammaTensor(
            child=out_child,
            min_vals=lazyrepeatarray(data=0, shape=out_child.shape),
            max_vals=lazyrepeatarray(
                data=self.max_vals.data - self.min_vals.data, shape=out_child.shape
            ),
            data_subjects=data_subjects,
            func_str=GAMMA_TENSOR_OP.PTP.value,
            sources=output_state,
        )

    def take(
        self,
        indices: ArrayLike,
        axis: Optional[int] = None,
        mode: str = "raise",
        out: Optional[np.ndarray] = None,
    ) -> GammaTensor:
        """Take elements from an array along an axis."""
        output_state = dict()
        output_state[self.id] = self
        out_child = self.child.take(indices, axis=axis, mode=mode, out=out)

        return GammaTensor(
            child=out_child,
            data_subjects=self.data_subjects.take(
                indices, axis=axis, mode=mode, out=out
            ),
            min_vals=lazyrepeatarray(data=self.min_vals.data, shape=out_child.shape),
            max_vals=lazyrepeatarray(data=self.max_vals.data, shape=out_child.shape),
            func_str=GAMMA_TENSOR_OP.TAKE.value,
            sources=output_state,
        )

    def put(
        self,
        ind: ArrayLike,
        v: ArrayLike,
        mode: str = "raise",
    ) -> GammaTensor:
        """Replaces specified elements of an array with given values.
        The indexing works on the flattened target array. put is roughly equivalent to:
            a.flat[ind] = v
        """
        output_state = dict()
        output_state[self.id] = self
        if self.min_vals.data > min(v) or self.max_vals.data < max(v):
            raise Exception("The v values must be within the data bounds")

        out_child = self.child
        out_child.put(ind, v, mode=mode)

        return GammaTensor(
            child=out_child,
            data_subjects=self.data_subjects,
            min_vals=self.min_vals,
            max_vals=self.max_vals,
            func_str=GAMMA_TENSOR_OP.PUT.value,
            sources=output_state,
        )

    def repeat(
        self, repeats: Union[int, Tuple[int, ...]], axis: Optional[int] = None
    ) -> GammaTensor:
        """
        Repeat elements of an array.

        Parameters
            repeats: int or array of ints

                The number of repetitions for each element. repeats is broadcasted to fit the shape of the given axis.

            axis: int, optional

                The axis along which to repeat values. By default, use the flattened input array, and return a flat
                output array.

        Returns

            repeated_array: PhiTensor

                Output array which has the same shape as a, except along the given axis.

        """
        sources = dict()
        sources[self.id] = self

        result = self.child.repeat(repeats, axis)
        if isinstance(self.min_vals, lazyrepeatarray):
            minv = lazyrepeatarray(data=self.min_vals.data.min(), shape=result.shape)
            maxv = lazyrepeatarray(data=self.max_vals.data.max(), shape=result.shape)
        else:
            minv = self.min_vals
            maxv = self.max_vals

        return GammaTensor(
            child=result,
            data_subjects=self.data_subjects.repeat(repeats, axis),
            min_vals=minv,
            max_vals=maxv,
            func_str=GAMMA_TENSOR_OP.REPEAT.value,
            sources=sources,
        )

    def cumsum(
        self,
        axis: Optional[int] = None,
    ) -> GammaTensor:
        """
        Return the cumulative sum of the elements along a given axis.

        Parameters
            axis: int, optional
                Axis along which the cumulative sum is computed. The default (None) is to compute the cumsum over the
                flattened array.
        Returns
            cumsum_along_axis: GammaTensor
                A new array holding the result is returned. The result has the same size as input, and the same shape as
                 a if axis is not None or a is 1-d.
        """
        result = self.child.cumsum(axis=axis)
        num = np.ones_like(self.child).cumsum(axis=axis)

        sources = dict()
        sources[self.id] = self

        return GammaTensor(
            child=result,
            data_subjects=self.data_subjects.cumsum(axis=axis),
            min_vals=lazyrepeatarray(
                data=(self.min_vals.data * num).min(), shape=result.shape
            ),
            max_vals=lazyrepeatarray(
                data=(self.max_vals.data * num).max(), shape=result.shape
            ),
            func_str=GAMMA_TENSOR_OP.CUMSUM.value,
            sources=sources,
        )

    def cumprod(
        self,
        axis: Optional[int] = None,
    ) -> GammaTensor:
        """
        Return the cumulative product of the elements along a given axis.

        Parameters
            axis: int, optional
                Axis along which the cumulative product is computed. The default (None) is to compute the cumprod over
                the flattened array.
        Returns
            cumprod_along_axis: GammaTensor
                A new array holding the result is returned. The result has the same size as input, and the same shape as
                 a if axis is not None or a is 1-d.
        """
        result = self.child.cumprod(axis=axis)
        num = np.ones_like(self.child).cumsum(axis=axis)
        if abs(self.max_vals.data) >= (self.min_vals.data):
            highest = abs(self.max_vals.data)
        else:
            highest = abs(self.min_vals.data)

        sources = dict()
        sources[self.id] = self

        return GammaTensor(
            child=result,
            data_subjects=self.data_subjects.cumprod(axis=axis),
            min_vals=lazyrepeatarray(data=-(highest**num).min(), shape=result.shape),
            max_vals=lazyrepeatarray(data=(highest**num).max(), shape=result.shape),
            func_str=GAMMA_TENSOR_OP.CUMPROD.value,
            sources=sources,
        )

    def prod(self, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> GammaTensor:
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
        sources = dict()
        sources[self.id] = self
        return GammaTensor(
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
            func_str=GAMMA_TENSOR_OP.PROD.value,
            sources=sources,
        )

    def __floordiv__(self, other: Any) -> GammaTensor:
        """
        return self // value.
        """
        # relative
        from .phi_tensor import PhiTensor

        sources = dict()
        sources[self.id] = self

        if isinstance(other, PhiTensor):
            return self // other.gamma
        elif isinstance(other, GammaTensor):
            sources[other.id] = other

            min_min = self.min_vals.data // other.min_vals.data
            min_max = self.min_vals.data // other.max_vals.data
            max_min = self.max_vals.data // other.min_vals.data
            max_max = self.max_vals.data // other.max_vals.data

            _min_vals = np.min([min_min, min_max, max_min, max_max], axis=0)  # type: ignore
            _max_vals = np.max([min_min, min_max, max_min, max_max], axis=0)  # type: ignore

            return GammaTensor(
                child=self.child // other.child,
                data_subjects=self.data_subjects,
                min_vals=lazyrepeatarray(data=_min_vals, shape=self.shape),
                max_vals=lazyrepeatarray(data=_max_vals, shape=self.shape),
                func_str=GAMMA_TENSOR_OP.FLOOR_DIVIDE.value,
                sources=sources,
            )
        elif is_acceptable_simple_type(other):
            sources["0"] = other
            return GammaTensor(
                child=self.child // other,
                data_subjects=self.data_subjects,
                min_vals=lazyrepeatarray(
                    data=self.min_vals.data // other, shape=self.min_vals.shape
                ),
                max_vals=lazyrepeatarray(
                    data=self.max_vals.data // other, shape=self.max_vals.shape
                ),
                func_str=GAMMA_TENSOR_OP.FLOOR_DIVIDE.value,
                sources=sources,
            )
        else:
            raise NotImplementedError(
                f"floordiv not supported between GammaTensor & {type(other)}"
            )

    def trace(self, offset: int = 0, axis1: int = 0, axis2: int = 1) -> GammaTensor:
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

            sum_along_diagonals: GammaTensor
                If a is 2-D, the sum along the diagonal is returned.
                If a has larger dimensions, then an array of sums along diagonals is returned.
        """

        sources = dict()
        sources[self.id] = self
        result = self.child.trace(offset, axis1, axis2)
        num = np.ones_like(self.child).trace(offset, axis1, axis2)
        return GammaTensor(
            child=result,
            data_subjects=self.data_subjects.trace(offset, axis1, axis2),
            min_vals=lazyrepeatarray(data=self.min_vals.data * num, shape=result.shape),
            max_vals=lazyrepeatarray(data=self.max_vals.data * num, shape=result.shape),
            func_str=GAMMA_TENSOR_OP.TRACE.value,
            sources=sources,
        )

    def diagonal(self, offset: int = 0, axis1: int = 0, axis2: int = 1) -> GammaTensor:
        """
        Return the sum along diagonals of the array.

        Return specified diagonals.
        If a is 2-D, returns the diagonal of a with the given offset, i.e., the collection of elements
        of the form a[i, i+offset].

        If a has more than two dimensions, then the axes specified by axis1 and axis are used to determine
        the 2-D sub-array whose diagonal is returned.  The shape of the resulting array can be determined by
        removing axis1 and axis2 and appending an index to the right equal to the size of the resulting diagonals.

        Parameters

            offset: int, optional
                Offset of the diagonal from the main diagonal.  Can be positive or negative.
                Defaults to main diagonal (0).
            axis1, axis2: int, optional
                Axis to be used as the first axis of the 2-D sub-arrays from which the diagonals should be taken.
                Defaults are the first two axes of a.

        Returns
            array_of_diagonals : Union[TensorWrappedPhiTensorPointer,MPCTensor]
                If a is 2-D, then a 1-D array containing the diagonal and of the same type as a is returned unless
                a is a matrix, in which case
                a 1-D array rather than a (2-D) matrix is returned in order to maintain backward compatibility.

                If a.ndim > 2, then the dimensions specified by axis1 and axis2 are removed, and a new axis
                inserted at the end corresponding to the diagonal.
        """
        sources = dict()
        sources[self.id] = self
        result = self.child.diagonal(offset, axis1, axis2)

        return GammaTensor(
            child=result,
            data_subjects=self.data_subjects.diagonal(offset, axis1, axis2),
            min_vals=lazyrepeatarray(data=self.min_vals.data, shape=result.shape),
            max_vals=lazyrepeatarray(data=self.max_vals.data, shape=result.shape),
            func_str=GAMMA_TENSOR_OP.DIAGONAL.value,
            sources=sources,
        )

    def min(
        self,
        axis: Optional[int] = None,
        keepdims: Optional[bool] = False,
        initial: Optional[float] = None,
        where: Optional[Union[List[bool], ArrayLike[bool]]] = None,
    ) -> GammaTensor:
        """
        Return the minimum of an array or minimum along an axis.

        Parameters
            axis: None or int or tuple of ints, optional
                Axis or axes along which to operate. By default, flattened input is used.
                If this is a tuple of ints, the minimum is selected over multiple axes,
                instead of a single axis or all the axes as before.

            keepdims: bool, optional
                If this is set to True, the axes which are reduced are left in the result as dimensions with size one.
                With this option, the result will broadcast correctly against the input array.
                If the default value is passed, then keepdims will not be passed through to the amin method of
                sub-classes of ndarray, however any non-default value will be.
                If the sub-class’ method does not implement keepdims any exceptions will be raised.
            initial: scalar, optional
                The maximum value of an output element. Must be present to allow computation on empty slice.
                See reduce for details.

            where: array_like of bool, optional
                Elements to compare for the minimum. See reduce for details.

        Returns
            a_min: GammaTensor
                Minimum of a.
                If axis is None, the result is a scalar value.
                If axis is given, the result is an array of dimension a.ndim - 1.
        """

        if where is None:
            sources = dict()
            sources[self.id] = self
            result = np.amin(self.child, axis=axis, keepdims=keepdims, initial=initial)
            indices = np.unravel_index(self.child.argmin(axis), shape=self.child.shape)

            return GammaTensor(
                child=result,
                data_subjects=self.data_subjects[indices],
                min_vals=lazyrepeatarray(data=self.min_vals.data, shape=result.shape),
                max_vals=lazyrepeatarray(data=self.max_vals.data, shape=result.shape),
                func_str=GAMMA_TENSOR_OP.MIN.value,
                sources=sources,
            )
        else:
            if initial is None:
                raise ValueError(
                    "reduction operation 'minimum' does not have an identity, "
                    "so to use a where mask one has to specify 'initial'"
                )
            else:
                sources = dict()
                sources[self.id] = self
                result = np.amin(
                    self.child,
                    axis=axis,
                    keepdims=keepdims,
                    initial=initial,
                    where=where,
                )
                indices = np.unravel_index(
                    self.child.argmin(axis), shape=self.child.shape
                )

                return GammaTensor(
                    child=result,
                    data_subjects=self.data_subjects[indices],
                    min_vals=lazyrepeatarray(
                        data=self.min_vals.data, shape=result.shape
                    ),
                    max_vals=lazyrepeatarray(
                        data=self.max_vals.data, shape=result.shape
                    ),
                    func_str=GAMMA_TENSOR_OP.MIN.value,
                    sources=sources,
                )

    def max(
        self,
        axis: Optional[int] = None,
        keepdims: Optional[bool] = False,
        initial: Optional[float] = None,
        where: Optional[Union[List[bool], ArrayLike[bool]]] = None,
    ) -> GammaTensor:
        """
        Return the maximum of an array or minimum along an axis.

        Parameters
            axis: None or int or tuple of ints, optional
                Axis or axes along which to operate. By default, flattened input is used.
                If this is a tuple of ints, the minimum is selected over multiple axes,
                instead of a single axis or all the axes as before.

            keepdims: bool, optional
                If this is set to True, the axes which are reduced are left in the result as dimensions with
                size one.
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
        if where is None:
            sources = dict()
            sources[self.id] = self
            result = np.amax(self.child, axis=axis, keepdims=keepdims, initial=initial)
            indices = np.unravel_index(self.child.argmax(axis), shape=self.child.shape)
            return GammaTensor(
                child=result,
                data_subjects=self.data_subjects[indices],
                min_vals=lazyrepeatarray(data=self.min_vals.data, shape=result.shape),
                max_vals=lazyrepeatarray(data=self.max_vals.data, shape=result.shape),
                func_str=GAMMA_TENSOR_OP.MAX.value,
                sources=sources,
            )
        else:
            if initial is None:
                raise ValueError(
                    "reduction operation 'minimum' does not have an identity, "
                    "so to use a where mask one has to specify 'initial'"
                )
            else:
                sources = dict()
                sources[self.id] = self
                result = np.amax(
                    self.child,
                    axis=axis,
                    keepdims=keepdims,
                    initial=initial,
                    where=where,
                )
                indices = np.unravel_index(
                    self.child.argmax(axis), shape=self.child.shape
                )
                return GammaTensor(
                    child=result,
                    data_subjects=self.data_subjects[indices],
                    min_vals=lazyrepeatarray(
                        data=self.min_vals.data, shape=result.shape
                    ),
                    max_vals=lazyrepeatarray(
                        data=self.max_vals.data, shape=result.shape
                    ),
                    func_str=GAMMA_TENSOR_OP.MAX.value,
                    sources=sources,
                )

    def __lshift__(self, other: Any) -> GammaTensor:
        # relative
        from .phi_tensor import PhiTensor

        sources = dict()
        sources[self.id] = self

        if is_acceptable_simple_type(other):
            sources["0"] = other
            child = self.child << other
            if isinstance(other, np.ndarray):
                other_max, other_min = other.max(), other.min()
            else:
                other_max, other_min = other, other
            data_subjects = self.data_subjects

        elif isinstance(other, PhiTensor):  # type: ignore
            return self << other.gamma
        elif isinstance(other, GammaTensor):
            sources[other.id] = other
            child = self.child << other.child
            other_max, other_min = other.max_vals.data, other.min_vals.data
            data_subjects = self.data_subjects + other.data_subjects
        else:
            raise NotImplementedError(
                f"lshift is not implemented for type: {type(other)}"
            )

        min_min = self.min_vals.data << other_min
        min_max = self.min_vals.data << other_max
        max_min = self.max_vals.data << other_min
        max_max = self.max_vals.data << other_max

        _min_vals = np.min([min_min, min_max, max_min, max_max], axis=0)  # type: ignore
        _max_vals = np.max([min_min, min_max, max_min, max_max], axis=0)  # type: ignore
        return GammaTensor(
            child=child,
            data_subjects=data_subjects,
            min_vals=lazyrepeatarray(data=_min_vals, shape=self.shape),
            max_vals=lazyrepeatarray(data=_max_vals, shape=self.shape),
            func_str=GAMMA_TENSOR_OP.LSHIFT.value,
            sources=sources,
        )

    def __rshift__(self, other: Any) -> GammaTensor:
        # relative
        from .phi_tensor import PhiTensor

        sources = dict()
        sources[self.id] = self

        if is_acceptable_simple_type(other):
            sources["0"] = other
            child = self.child >> other
            if isinstance(other, np.ndarray):
                other_max, other_min = other.max(), other.min()
            else:
                other_max, other_min = other, other
            data_subjects = self.data_subjects

        elif isinstance(other, PhiTensor):  # type: ignore
            return self >> other.gamma
        elif isinstance(other, GammaTensor):
            sources[other.id] = other
            child = self.child >> other.child
            other_max, other_min = other.max_vals.data, other.min_vals.data
            data_subjects = self.data_subjects + other.data_subjects
        else:
            raise NotImplementedError(
                f"rshift is not implemented for type: {type(other)}"
            )

        min_min = self.min_vals.data >> other_min
        min_max = self.min_vals.data >> other_max
        max_min = self.max_vals.data >> other_min
        max_max = self.max_vals.data >> other_max

        _min_vals = np.min([min_min, min_max, max_min, max_max], axis=0)  # type: ignore
        _max_vals = np.max([min_min, min_max, max_min, max_max], axis=0)  # type: ignore

        return GammaTensor(
            child=child,
            data_subjects=data_subjects,
            min_vals=lazyrepeatarray(data=_min_vals, shape=self.shape),
            max_vals=lazyrepeatarray(data=_max_vals, shape=self.shape),
            func_str=GAMMA_TENSOR_OP.RSHIFT.value,
            sources=sources,
        )

    def __xor__(self, other: Any) -> GammaTensor:
        # relative
        from .phi_tensor import PhiTensor

        sources = dict()
        sources[self.id] = self

        if is_acceptable_simple_type(other):
            sources["0"] = other
            child = self.child ^ other
            data_subjects = self.data_subjects
            if isinstance(other, np.ndarray):
                other_min, other_max = other.min(), other.max()
            else:
                other_min, other_max = other, other
        elif isinstance(other, PhiTensor):
            return self ^ other.gamma
        elif isinstance(other, GammaTensor):
            sources[other.id] = other
            child = self.child ^ other.child
            data_subjects = self.data_subjects + other.data_subjects
            other_min, other_max = other.min_vals.data, other.max_vals.data
        else:
            raise NotImplementedError(f"xor is not implemented for type: {type(other)}")

        # TODO: should modify for a tighter found for xor
        _max = int(max(self.max_vals.data, other_max))
        _min = int(min(self.min_vals.data, other_min))
        _max_vals = max(
            (2 ** (_min ^ _max).bit_length()) - 1, (2 ** (_max).bit_length()) - 1
        )
        _min_vals = min(0, _min)

        return GammaTensor(
            child=child,
            data_subjects=data_subjects,
            min_vals=lazyrepeatarray(data=_min_vals, shape=self.shape),
            max_vals=lazyrepeatarray(data=_max_vals, shape=self.shape),
            func_str=GAMMA_TENSOR_OP.XOR.value,
            sources=sources,
        )

    def __round__(self, n: int = 0) -> GammaTensor:
        sources = dict()
        sources[self.id] = self

        sources["0"] = n  # type: ignore
        child = self.child.round(n)

        return GammaTensor(
            child=child,
            data_subjects=self.data_subjects,
            min_vals=lazyrepeatarray(
                data=self.min_vals.data.round(n), shape=self.min_vals.shape
            ),
            max_vals=lazyrepeatarray(
                data=self.max_vals.data.round(n), shape=self.max_vals.shape
            ),
            func_str=GAMMA_TENSOR_OP.ROUND.value,
            sources=sources,
        )

    def round(self, n: int = 0) -> GammaTensor:
        return self.__round__(n)

    def sort(self, axis: int = -1, kind: Optional[str] = None) -> GammaTensor:
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
        sources = dict()
        sources[self.id] = self

        out_ds = np.take_along_axis(self.data_subjects, indices, axis=axis)
        return GammaTensor(
            child=self.child,
            data_subjects=out_ds,
            min_vals=self.min_vals,
            max_vals=self.max_vals,
            func_str=GAMMA_TENSOR_OP.SORT.value,
            sources=sources,
        )

    def argsort(self, axis: Optional[int] = -1) -> GammaTensor:
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
        sources = dict()
        sources[self.id] = self
        result = self.child.argsort(axis)
        out_ds = np.take_along_axis(self.data_subjects, result, axis=axis)
        return GammaTensor(
            child=result,
            data_subjects=out_ds,
            min_vals=lazyrepeatarray(data=0, shape=self.shape),
            max_vals=lazyrepeatarray(data=self.child.size, shape=self.shape),
            func_str=GAMMA_TENSOR_OP.ARGSORT.value,
            sources=sources,
        )

    def choose(
        self,
        choices: Union[Sequence, np.ndarray, PassthroughTensor],
        mode: Optional[str] = "raise",
    ) -> GammaTensor:
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
        # relative
        from .phi_tensor import PhiTensor

        sources = dict()
        sources[self.id] = self
        if isinstance(choices, PhiTensor):
            choices = choices.gamma

        if isinstance(choices, GammaTensor):
            sources[choices.id] = choices
            result = np.choose(self.child, choices.child, mode=mode)
            output_ds = np.choose(self.child, choices.data_subjects)
        else:
            raise NotImplementedError(
                f"Object type: {type(choices)} This leads to a data leak or side channel attack"
            )

        return GammaTensor(
            child=result,
            data_subjects=output_ds,
            min_vals=lazyrepeatarray(
                data=choices.min_vals.data.min(), shape=result.shape
            ),
            max_vals=lazyrepeatarray(
                data=choices.max_vals.data.max(), shape=result.shape
            ),
            sources=sources,
            func_str=GAMMA_TENSOR_OP.CHOOSE.value,
        )

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.child.shape

    @property
    def lipschitz_bound(self) -> float:
        # TODO: Check if there are any functions for which lipschitz bounds shouldn't be computed
        # if dis(self.func) == dis(no_op):
        #     raise Exception

        print("Starting JAX JIT")
        # relative
        from .gamma_functions import GAMMA_FUNC_MAPPER

        fn = jax.jit(GAMMA_FUNC_MAPPER[GAMMA_TENSOR_OP(self.func_str)])
        print("Traced self.func with jax's jit, now calculating gradient")
        grad_fn = jax.grad(fn)
        print("Obtained gradient, creating lookup tables")
        i2k, k2i, i2v, i2s = create_new_lookup_tables(self.sources)

        print("created lookup tables, now getting bounds")
        i2minval = jnp.concatenate([x for x in i2v]).reshape(-1, 1)
        i2maxval = jnp.concatenate([x for x in i2v]).reshape(-1, 1)
        bounds = jnp.concatenate([i2minval, i2maxval], axis=1)
        print("Obtained bounds")
        # sample_input = i2minval.reshape(-1)
        _ = i2minval.reshape(-1)
        print("Obtained all inputs")

        def max_grad_fn(input_values: np.ndarray) -> float:
            vectors = {}
            n = 0
            for i, size_param in enumerate(i2s):
                vectors[i2k[i]] = input_values[n : n + size_param]  # noqa: E203
                n += size_param

            grad_pred = grad_fn(vectors)

            m = 0
            for value in grad_pred.values():
                m = max(m, jnp.max(value))

            # return negative because we want to maximize instead of minimize
            return -m

        print("starting SHGO")
        res = shgo(max_grad_fn, bounds, iters=1, constraints=tuple())
        print("Ran SHGO")
        # return negative because we flipped earlier
        return -float(res.fun)

    @property
    def dtype(self) -> np.dtype:
        return self.child.dtype

    @staticmethod
    def get_input_tensors(state_tree: dict[int, GammaTensor]) -> List:
        # TODO: See if we can call np.stack on the output and create a vectorized tensor instead of a list of tensors
        input_tensors = []
        for tensor in state_tree.values():
            if tensor.func_str == GAMMA_TENSOR_OP.NOOP.value:
                input_tensors.append(tensor)
            else:
                input_tensors += GammaTensor.get_input_tensors(tensor.sources)
        return input_tensors

    def _object2bytes(self) -> bytes:
        schema = get_capnp_schema(schema_file="gamma_tensor.capnp")

        gamma_tensor_struct: CapnpModule = schema.GammaTensor  # type: ignore
        gamma_msg = gamma_tensor_struct.new_message()
        # this is how we dispatch correct deserialization of bytes
        gamma_msg.magicHeader = serde_magic_header(type(self))

        # do we need to serde func? if so how?
        # what about the state dict?

        if isinstance(self.child, np.ndarray) or np.isscalar(self.child):
            chunk_bytes(capnp_serialize(np.array(self.child), to_bytes=True), "child", gamma_msg)  # type: ignore
            gamma_msg.isNumpy = True
        elif isinstance(self.child, jnp.ndarray):
            chunk_bytes(
                capnp_serialize(jax2numpy(self.child, self.child.dtype), to_bytes=True),
                "child",
                gamma_msg,
            )
            gamma_msg.isNumpy = True
        else:
            chunk_bytes(serialize(self.child, to_bytes=True), "child", gamma_msg)  # type: ignore
            gamma_msg.isNumpy = False

        gamma_msg.sources = serialize(self.sources, to_bytes=True)
        chunk_bytes(
            capnp_serialize(dslarraytonumpyutf8(self.data_subjects), to_bytes=True),
            "dataSubjects",
            gamma_msg,
        )

        # Explicity convert lazyrepeatarray data to ndarray
        self.min_vals.data = np.array(self.min_vals.data)
        self.max_vals.data = np.array(self.max_vals.data)

        gamma_msg.minVal = serialize(self.min_vals, to_bytes=True)
        gamma_msg.maxVal = serialize(self.max_vals, to_bytes=True)
        gamma_msg.isLinear = self.is_linear
        gamma_msg.id = self.id
        gamma_msg.funcStr = self.func_str

        # return gamma_msg.to_bytes_packed()
        return gamma_msg.to_bytes()

    @staticmethod
    def _bytes2object(buf: bytes) -> GammaTensor:
        schema = get_capnp_schema(schema_file="gamma_tensor.capnp")
        gamma_struct: CapnpModule = schema.GammaTensor  # type: ignore
        # https://stackoverflow.com/questions/48458839/capnproto-maximum-filesize
        MAX_TRAVERSAL_LIMIT = 2**64 - 1
        # capnp from_bytes is now a context
        with gamma_struct.from_bytes(
            buf, traversal_limit_in_words=MAX_TRAVERSAL_LIMIT
        ) as gamma_msg:

            if gamma_msg.isNumpy:
                child = capnp_deserialize(
                    combine_bytes(gamma_msg.child), from_bytes=True
                )
            else:
                child = deserialize(combine_bytes(gamma_msg.child), from_bytes=True)

            state = deserialize(gamma_msg.sources, from_bytes=True)

            data_subjects = numpyutf8todslarray(
                capnp_deserialize(
                    combine_bytes(gamma_msg.dataSubjects), from_bytes=True
                )
            )

            min_val = deserialize(gamma_msg.minVal, from_bytes=True)
            max_val = deserialize(gamma_msg.maxVal, from_bytes=True)
            is_linear = gamma_msg.isLinear
            id_str = gamma_msg.id
            func_str = gamma_msg.funcStr

            return GammaTensor(
                child=child,
                data_subjects=data_subjects,
                min_vals=min_val,
                max_vals=max_val,
                is_linear=is_linear,
                sources=state,
                id=id_str,
                func_str=func_str,
            )
