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
from numpy.typing import NDArray
from scipy.ndimage.interpolation import rotate

# relative
from .... import lib
from ....ast.klass import pointerize_args_and_kwargs
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


@serializable(recursive_serde=True)
class TensorWrappedPhiTensorPointer(Pointer, PassthroughTensor):
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
    }
    _exhausted = False
    is_enum = False

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
            * (self.max_vals.to_numpy() - self.min_vals.to_numpy())
            + self.min_vals.to_numpy()
        ).astype(public_dtype_func())

    def __repr__(self) -> str:
        return (
            self.synthetic.__repr__()
            + "\n\n (The data printed above is synthetic - it's an imitation of the real data.)"
        )

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
            if (self.data_subjects != other.data_subjects).all():  # type: ignore
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
        """Apply the "matmul" operation between "self" and "other"

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

    def concatenate(
        self,
        other: TensorWrappedPhiTensorPointer,
        *args: List[Any],
        **kwargs: Dict[str, Any],
    ) -> MPCTensor:
        """Apply the "add" operation between "self" and "other"

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

    def sum(
        self,
        *args: Tuple[Any, ...],
        **kwargs: Any,
    ) -> Union[
        TensorWrappedPhiTensorPointer, MPCTensor, TensorWrappedGammaTensorPointer
    ]:
        """Apply the "truediv" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedPhiTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedPhiTensorPointer,MPCTensor] : Result of the operation.
        """
        return self.gamma.sum(*args, **kwargs)

    def __getitem__(
        self, key: Union[int, bool, slice]
    ) -> TensorWrappedPhiTensorPointer:
        """Apply the slice  operation on "self"
        Args:
            y (Union[int,bool,slice]) : second operand.

        Returns:
            Union[TensorWrappedPhiTensorPointer,MPCTensor] : Result of the operation.
        """
        attr_path_and_name = "syft.core.tensor.tensor.Tensor.__getitem__"
        result: TensorWrappedPhiTensorPointer
        min_vals = self.min_vals.__getitem__(key)
        max_vals = self.max_vals.__getitem__(key)

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
            ) = lib.python.util.downcast_args_and_kwargs(args=[key], kwargs={})

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
            args=[key],
            kwargs={},
        )
        dummy_res = np.empty(self.public_shape).__getitem__(key)
        result.public_shape = dummy_res.shape
        result.public_dtype = self.public_dtype

        return result

    def ones_like(
        self,
        *args: Tuple[Any, ...],
        **kwargs: Any,
    ) -> TensorWrappedPhiTensorPointer:
        """Apply the "truediv" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedPhiTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedPhiTensorPointer,MPCTensor] : Result of the operation.
        """
        attr_path_and_name = "syft.core.tensor.tensor.Tensor.ones_like"
        result: TensorWrappedPhiTensorPointer
        min_vals = self.min_vals.ones_like(*args, **kwargs)
        max_vals = self.max_vals.ones_like(*args, **kwargs)

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
        dummy_res = np.ones_like(np.empty(self.public_shape), *args, **kwargs)
        result.public_shape = dummy_res.shape
        result.public_dtype = self.public_dtype

        return result

    def exp(
        self,
    ) -> Union[TensorWrappedPhiTensorPointer, MPCTensor]:
        """Apply the "truediv" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedPhiTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedPhiTensorPointer,MPCTensor] : Result of the operation.
        """
        attr_path_and_name = "syft.core.tensor.tensor.Tensor.exp"

        # TODO: should modify to log reduction.
        def exp_reduction(val: np.ndarray) -> np.ndarray:
            pos_index = val >= 0
            neg_index = val < 0
            exp = np.exp((pos_index * val * -1) + (neg_index * val))
            pos_values = (pos_index) * exp
            neg_values = (neg_index) * exp * -1
            return pos_values + neg_values

        min_vals = self.min_vals.copy()
        min_vals.data = np.array(exp_reduction(min_vals.data))
        max_vals = self.max_vals.copy()
        max_vals.data = np.array(exp_reduction(max_vals.data))

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
            ) = lib.python.util.downcast_args_and_kwargs(args=[], kwargs={})

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

        result.public_shape = self.public_shape
        result.public_dtype = self.public_dtype

        return result

    def reciprocal(
        self,
    ) -> Union[TensorWrappedPhiTensorPointer, MPCTensor]:
        """Apply the "reciprocal" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedPhiTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedPhiTensorPointer,MPCTensor] : Result of the operation.
        """
        attr_path_and_name = "syft.core.tensor.tensor.Tensor.reciprocal"

        min_vals = self.min_vals.copy()
        min_vals.data = np.array(1 / min_vals.data)
        max_vals = self.max_vals.copy()
        max_vals.data = np.array(1 / max_vals.data)

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
            ) = lib.python.util.downcast_args_and_kwargs(args=[], kwargs={})

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

        result.public_shape = self.public_shape
        result.public_dtype = self.public_dtype

        return result

    def softmax(
        self,
    ) -> Union[TensorWrappedPhiTensorPointer, MPCTensor]:
        """Apply the "softmax" operation on "self"

        Args:
            y (Union[TensorWrappedPhiTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedPhiTensorPointer,MPCTensor] : Result of the operation.
        """
        attr_path_and_name = "syft.core.tensor.tensor.Tensor.softmax"

        # TODO: should modify to log reduction.
        def softmax(val: np.ndarray) -> np.ndarray:
            logits = val - val.max()
            numerator = np.exp(logits)
            inv = 1 / numerator.sum()
            return numerator * inv

        min_vals = self.min_vals.copy()
        min_vals.data = np.array(softmax(min_vals.data))
        max_vals = self.max_vals.copy()
        max_vals.data = np.array(softmax(max_vals.data))

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
            ) = lib.python.util.downcast_args_and_kwargs(args=[], kwargs={})

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

        result.public_shape = self.public_shape
        result.public_dtype = self.public_dtype

        return result

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

    def one_hot(self: TensorWrappedPhiTensorPointer) -> np.array:
        tensor_size = np.empty(self.public_shape).size
        one_hot_Y = np.zeros((tensor_size, self.max_vals.data[0] + 1))
        one_hot_Y = one_hot_Y.T

        attr_path_and_name = "syft.core.tensor.tensor.Tensor.one_hot"

        result = TensorWrappedPhiTensorPointer(
            data_subjects=self.data_subjects,
            min_vals=self.min_vals,
            max_vals=self.max_vals,
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

        result.public_shape = one_hot_Y.shape
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


@implements(TensorWrappedPhiTensorPointer, np.ones_like)
def ones_like(
    tensor: TensorWrappedPhiTensorPointer,
    *args: Tuple[Any, ...],
    **kwargs: Dict[Any, Any],
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
            data_subjects=self.data_subjects.copy(),
        )

    def all(self) -> bool:
        return self.child.all()

    def any(self) -> bool:
        return self.child.any()

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
        *args: Tuple[Any, ...],
        **kwargs: Any,
    ) -> Union[PhiTensor, GammaTensor]:
        # TODO: Add support for axes arguments later
        min_vals = self.min_vals.zeros_like(*args, **kwargs)
        max_vals = self.max_vals.zeros_like(*args, **kwargs)

        child = (
            np.zeros_like(self.child, *args, **kwargs)
            if isinstance(self.child, np.ndarray)
            else self.child.ones_like(*args, **kwargs)
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

    def abs(self) -> PhiTensor:
        data = self.child
        output = np.abs(data)

        return PhiTensor(
            child=output,
            data_subjects=self.data_subjects,
            min_vals=np.abs(self.min_vals.data),
            max_vals=np.abs(self.min_vals.data),
        )

    def reshape(self, *shape: Tuple[int, ...]) -> PhiTensor:

        data = self.child
        output_data = np.reshape(data, *shape)

        return PhiTensor(
            child=output_data,
            data_subjects=np.reshape(self.data_subjects, *shape),
            min_vals=output_data.min(),
            max_vals=output_data.max(),
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

    def ravel(self) -> PhiTensor:
        data = self.child
        output_data = data.ravel()

        output_data_subjects = self.data_subjects.ravel()

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
        indices = self.child.argmax(axis)
        result = self.child.max(axis)
        return PhiTensor(
            child=result,
            data_subjects=self.data_subjects[indices],
            min_vals=lazyrepeatarray(data=result.min(), shape=result.shape),
            max_vals=lazyrepeatarray(data=result.max(), shape=result.shape),
        )

    def min(self, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> PhiTensor:
        indices = self.child.argmin(axis)
        result = self.child.min(axis)
        return PhiTensor(
            child=result,
            data_subjects=self.data_subjects[indices],
            min_vals=lazyrepeatarray(data=result.min(), shape=result.shape),
            max_vals=lazyrepeatarray(data=result.max(), shape=result.shape),
        )

    def _argmax(self, axis: Optional[int]) -> PhiTensor:
        return self.child.argmax(axis)

    def unravel_argmax(
        self, axis: Optional[int] = None
    ) -> Tuple[np.ndarray]:  # possible privacy violation?
        arg_result = self._argmax(axis=axis)
        shape = self.shape
        return np.unravel_index(arg_result, shape)

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
        result = self.child.std(axis, **kwargs)
        return PhiTensor(
            child=result,
            data_subjects=self.data_subjects.std(axis, **kwargs),
            min_vals=lazyrepeatarray(data=0, shape=result.shape),
            max_vals=lazyrepeatarray(
                data=0.25
                * (self.max_vals.data - self.min_vals.data)
                ** 2,  # rough approximation, could be off
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

    def view(self, *args: List[Any]) -> PhiTensor:
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
            if (self.data_subjects == other.data_subjects).all():
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
            if (self.data_subjects != other.data_subjects).all():
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
            # Modify before merge, to know is broadcast is actually necessary
            if False:  # and not is_broadcastable(self.shape, other.shape):
                raise Exception(
                    f"Shapes not broadcastable: {self.shape} and {other.shape}"
                )
            else:
                if isinstance(other, np.ndarray):
                    data = self.child.__matmul__(other)
                    min_vals = self.min_vals.__matmul__(other)
                    max_vals = self.max_vals.__matmul__(other)
                    output_ds = self.data_subjects @ other
                elif isinstance(other, PhiTensor):
                    return self.gamma @ other.gamma
                    # if self.data_subjects != other.data_subjects:
                    #     return self.gamma @ other.gamma
                    # else:
                    #     data = self.child.__matmul__(other.child)
                    #     min_vals = self.min_vals.__matmul__(other.min_vals)
                    #     max_vals = self.max_vals.__matmul__(other.max_vals)
                    #     output_ds = DataSubjectList(
                    #         one_hot_lookup=np.concatenate(
                    #             (
                    #                 self.data_subjects.one_hot_lookup,
                    #                 other.data_subjects.one_hot_lookup,
                    #             )
                    #         ),
                    #         data_subjects_indexed=np.concatenate(
                    #             (np.zeros_like(data), np.ones_like(data))
                    #         ),  # replace with (1, *data.shape) if inc shape
                    #     )

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
        data: np.ndarray
        if (
            isinstance(self.child, int)
            or isinstance(self.child, float)
            or isinstance(self.child, bool)
        ):
            # For these data types, the transpose operation is meaningless, so don't change them.
            data = self.child  # type: ignore
            print(
                f"Warning: Tensor data was of type {type(data)}, transpose operation had no effect."
            )
        else:
            data = self.child.transpose(*args)

        # TODO: Should we give warnings for min_vals and max_vals being single floats/integers/booleans too?
        if (
            isinstance(self.min_vals, int)
            or isinstance(self.min_vals, float)
            or isinstance(self.min_vals, bool)
        ):
            # For these data types, the transpose operation is meaningless, so don't change them.
            min_vals = self.min_vals
            # print(f'Warning: Tensor data was of type {type(data)}, transpose operation had no effect.')
        else:
            min_vals = data.min()

        if (
            isinstance(self.max_vals, int)
            or isinstance(self.max_vals, float)
            or isinstance(self.max_vals, bool)
        ):
            # For these data types, the transpose operation is meaningless, so don't change them.
            max_vals = self.max_vals
            # print(f'Warning: Tensor data was of type {type(data)}, transpose operation had no effect.')
        else:
            max_vals = data.max()

        output_ds = self.data_subjects.transpose(*args)

        return PhiTensor(
            child=data,
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
            data_subjects=self.data_subjects.reshape(output_data.shape),
            min_vals=lazyrepeatarray(data=self.min_vals.data, shape=output_data.shape),
            max_vals=lazyrepeatarray(data=self.max_vals.data, shape=output_data.shape),
        )

    def concatenate(
        self,
        other: Union[np.ndarray, PhiTensor],
        *args: List[Any],
        **kwargs: Dict[str, Any],
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
            if (self.data_subjects == other.data_subjects).all():
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
            if (self.data_subjects == other.data_subjects).all():
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
            if (self.data_subjects == other.data_subjects).all():
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
            if (self.data_subjects == other.data_subjects).all():
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
        **kwargs: Any,
    ) -> Union[PhiTensor, GammaTensor]:
        return self.gamma.sum(axis, **kwargs)
        # # TODO: Add support for axes arguments later
        # min_val = self.min_vals.sum(axis=axis)
        # max_val = self.max_vals.sum(axis=axis)
        # if len(self.data_subjects.one_hot_lookup) == 1:
        #     result = self.child.sum(axis=axis)
        #     return PhiTensor(
        #         child=result,
        #         min_vals=min_val,
        #         max_vals=max_val,
        #         data_subjects=self.data_subjects.sum(target_shape=result.shape),
        #     )
        # result = self.child.sum(axis=axis)
        # return GammaTensor(
        #     child=result,
        #     data_subjects=self.data_subjects.sum(target_shape=result.shape),
        #     min_vals=min_val,
        #     max_vals=max_val,
        # )

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
        *args: Tuple[Any, ...],
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

    def exp(self) -> PhiTensor:
        # relative
        from ...smpc.approximations import exp

        def exp_reduction(val: np.ndarray) -> np.ndarray:
            pos_index = val >= 0
            neg_index = val < 0
            exp = np.exp((pos_index * val * -1) + (neg_index * val))
            pos_values = (pos_index) * exp
            neg_values = (neg_index) * exp * -1
            return pos_values + neg_values

        min_vals = self.min_vals.copy()
        min_vals.data = np.array(exp_reduction(min_vals.data))
        max_vals = self.max_vals.copy()
        max_vals.data = np.array(exp_reduction(max_vals.data))

        return PhiTensor(
            child=exp(self.child),  # type: ignore
            min_vals=min_vals,
            max_vals=max_vals,
            data_subjects=self.data_subjects,
        )

    def softmax(self) -> PhiTensor:
        # relative
        from ...smpc.approximations import exp
        from ...smpc.approximations import reciprocal

        def softmax(val: np.ndarray) -> np.ndarray:
            logits = val - val.max()
            numerator = np.exp(logits)
            inv = 1 / numerator.sum()
            return numerator * inv

        min_vals = self.min_vals.copy()
        min_vals.data = np.array(softmax(min_vals.data))
        max_vals = self.max_vals.copy()
        max_vals.data = np.array(softmax(max_vals.data))
        fpt = self.child.copy()
        if not isinstance(fpt.child, np.ndarray):
            raise ValueError("Softmax currently works only for numpy child")

        fpt.child = fpt.child - fpt.child.max()
        numerator = exp(fpt)
        inv = reciprocal(numerator.sum())  # type: ignore

        return PhiTensor(
            child=numerator * inv,  # type: ignore
            min_vals=min_vals,
            max_vals=max_vals,
            data_subjects=self.data_subjects,
        )

    def reciprocal(self) -> PhiTensor:
        # relative
        from ...smpc.approximations import reciprocal

        min_vals = self.min_vals.copy()
        min_vals.data = np.array(1 / min_vals.data)
        max_vals = self.max_vals.copy()
        max_vals.data = np.array(1 / max_vals.data)

        return PhiTensor(
            child=reciprocal(self.child),
            min_vals=min_vals,
            max_vals=max_vals,
            data_subjects=self.data_subjects,
        )

    def one_hot(self) -> PhiTensor:
        one_hot_child = self.child.one_hot()

        return PhiTensor(
            child=one_hot_child,
            min_vals=self.min_vals,
            max_vals=self.max_vals,
            data_subjects=self.data_subjects,
        )

    def resize(self, new_shape: Union[int, Tuple[int, ...]]) -> PhiTensor:
        out_child = np.resize(self.child.data, new_shape)
        return PhiTensor(
            child=out_child,
            min_vals=lazyrepeatarray(data=self.min_vals.data, shape=out_child.shape),
            max_vals=lazyrepeatarray(data=self.max_vals.data, shape=out_child.shape),
            data_subjects=np.resize(self.data_subjects, new_shape),
        )

    def compress(self, condition: List[bool], axis: Optional[int] = None) -> PhiTensor:
        out_child = self.child.compress(condition, axis)
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
        choices: Sequence[Union[PassthroughTensor, np.ndarray]],
        out: Optional[np.ndarray] = None,
        mode: Optional[str] = "raise",
    ) -> PhiTensor:
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
        result = self.child.choose(choices, mode=mode)
        if isinstance(self.min_vals, lazyrepeatarray):
            minv = lazyrepeatarray(data=self.min_vals.data.min(), shape=result.shape)
            maxv = lazyrepeatarray(data=self.max_vals.data.max(), shape=result.shape)
        else:
            minv, maxv = self.min_vals, self.max_vals

        return PhiTensor(
            child=result,
            data_subjects=self.data_subjects.take(choices),
            min_vals=minv,
            max_vals=maxv,
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
