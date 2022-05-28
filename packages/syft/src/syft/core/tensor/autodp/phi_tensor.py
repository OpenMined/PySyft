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

# relative
from .... import lib
from ....ast.klass import pointerize_args_and_kwargs
from ....core.adp.data_subject import DataSubject
from ....core.adp.data_subject_ledger import DataSubjectLedger
from ....core.adp.data_subject_list import DataSubjectList
from ....core.adp.data_subject_list import liststrtonumpyutf8
from ....core.adp.data_subject_list import numpyutf8tolist
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
from ..smpc.mpc_tensor import ShareTensor
from ..smpc.utils import TYPE_TO_RING_SIZE
from .adp_tensor import ADPTensor
from .gamma_tensor import GammaTensor
from .gamma_tensor import TensorWrappedGammaTensorPointer


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
    }
    _exhausted = False
    is_enum = False

    def __init__(
        self,
        data_subjects: DataSubjectList,
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
            if self.data_subjects != other.data_subjects:
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
    ) -> Union[
        TensorWrappedPhiTensorPointer, MPCTensor, TensorWrappedGammaTensorPointer
    ]:
        """Apply the "truediv" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedPhiTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedPhiTensorPointer,MPCTensor] : Result of the operation.
        """
        attr_path_and_name = "syft.core.tensor.tensor.Tensor.sum"
        result: Union[TensorWrappedGammaTensorPointer, TensorWrappedPhiTensorPointer]
        min_vals, max_vals = compute_min_max(self.min_vals, self.max_vals, None, "sum")
        if len(self.data_subjects.one_hot_lookup) == 1:
            result = TensorWrappedPhiTensorPointer(
                data_subjects=self.data_subjects,
                min_vals=min_vals,
                max_vals=max_vals,
                client=self.client,
            )
        else:
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

        result.public_shape = np.array([1]).shape
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
                child=FixedPrecisionTensor(value=None),
                data_subjects=self.data_subjects,
                min_vals=self.min_vals,  # type: ignore
                max_vals=self.max_vals,  # type: ignore
            ),
            public_shape=public_shape,
            public_dtype=public_dtype,
        )


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
        child: Sequence,
        data_subjects: Union[List[DataSubject], DataSubjectList],
        min_vals: np.ndarray,
        max_vals: np.ndarray,
    ) -> None:
        if isinstance(child, FixedPrecisionTensor):
            # child = the actual private data
            super().__init__(child)
        else:
            super().__init__(FixedPrecisionTensor(value=child))

        # lazyrepeatarray matching the shape of child
        if not isinstance(min_vals, lazyrepeatarray):
            min_vals = lazyrepeatarray(data=min_vals, shape=child.shape)  # type: ignore
        if not isinstance(max_vals, lazyrepeatarray):
            max_vals = lazyrepeatarray(data=max_vals, shape=child.shape)  # type: ignore
        self.min_vals = min_vals
        self.max_vals = max_vals

        if not isinstance(data_subjects, DataSubjectList):
            data_subjects = DataSubjectList.from_objs(data_subjects)

        self.data_subjects = data_subjects

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
            return PhiTensor(
                child=self.child.getitem(item.child),
                min_vals=self.min_vals,
                max_vals=self.max_vals,
                data_subjects=self.data_subjects,
            )
        else:
            return PhiTensor(
                child=self.child.getitem(item),
                min_vals=self.min_vals,
                max_vals=self.max_vals,
                data_subjects=self.data_subjects,
            )

    def create_gamma(self) -> GammaTensor:
        """Return a new Gamma tensor based on this phi tensor"""
        # TODO: check if values needs to be a JAX array or if numpy will suffice
        fpt_values = self.child

        gamma_tensor = GammaTensor(
            child=self.child,
            data_subjects=self.data_subjects,
            min_val=self.min_vals,
            max_val=self.max_vals,
            fpt_values=fpt_values,
        )

        return gamma_tensor

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
        gamma.state[gamma.id] = gamma

        res = gamma.publish(
            get_budget_for_user=get_budget_for_user,
            deduct_epsilon_for_user=deduct_epsilon_for_user,
            ledger=ledger,
            sigma=sigma,
        )
        fpt_values = gamma.fpt_values

        if fpt_values is None:
            raise ValueError(
                "FixedPrecisionTensor values should not be None after publish"
            )

        if isinstance(fpt_values.child, ShareTensor):
            fpt_values.child.child = res
        else:
            fpt_values.child = res

        print("Final FPT Values", fpt_values)

        return fpt_values

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
        return self.child.shape

    def __repr__(self) -> str:
        """Pretty print some information, optimized for Jupyter notebook viewing."""
        return (
            f"{self.__class__.__name__}(child={self.child}, "
            + f"min_vals={self.min_vals}, max_vals={self.max_vals})"
        )

    def __eq__(self, other: Any) -> Union[PhiTensor, GammaTensor]:  # type: ignore
        # TODO: what about data_subjects and min / max values?
        if is_acceptable_simple_type(other) or len(self.child) == len(other.child):
            gamma_output = False
            if is_acceptable_simple_type(other):
                result = self.child == other
            else:
                # check data_subjects match, if they dont gamma_output = True
                result = self.child == other.child
                if isinstance(result, GammaTensor):  # TODO: Check this
                    gamma_output = True
            if not gamma_output:
                # min_vals=self.min_vals * 0.0,
                # max_vals=self.max_vals * 0.0 + 1.0,
                return self.copy_with(child=result)
            else:
                return self.copy_with(child=result).gamma
        else:
            raise Exception(
                "Tensor dims do not match for __eq__: "
                + f"{len(self.child)} != {len(other.child)}"
            )

    def __add__(self, other: SupportedChainType) -> Union[PhiTensor, GammaTensor]:

        # if the tensor being added is also private
        if isinstance(other, PhiTensor):
            if self.data_subjects != other.data_subjects:
                return self.gamma + other.gamma

            return PhiTensor(
                child=self.child + other.child,
                min_vals=self.min_vals + other.min_vals,
                max_vals=self.max_vals + other.max_vals,
                data_subjects=self.data_subjects,
                # scalar_manager=self.scalar_manager,
            )

        # if the tensor being added is a public tensor / int / float / etc.
        elif is_acceptable_simple_type(other):
            return PhiTensor(
                child=self.child + other,
                min_vals=self.min_vals + other,
                max_vals=self.max_vals + other,
                data_subjects=self.data_subjects,
                # scalar_manager=self.scalar_manager,
            )

        elif isinstance(other, GammaTensor):
            return self.gamma + other
        else:
            print("Type is unsupported:" + str(type(other)))
            raise NotImplementedError

    def __sub__(self, other: SupportedChainType) -> Union[PhiTensor, GammaTensor]:

        if isinstance(other, PhiTensor):
            if self.data_subjects != other.data_subjects:
                # return self.gamma - other.gamma
                raise NotImplementedError

            data = self.child - other.child

            min_min = self.min_vals.data - other.min_vals.data
            min_max = self.min_vals.data - other.max_vals.data
            max_min = self.max_vals.data - other.min_vals.data
            max_max = self.max_vals.data - other.max_vals.data
            _min_vals = np.minimum.reduce([min_min, min_max, max_min, max_max])
            _max_vals = np.maximum.reduce([min_min, min_max, max_min, max_max])
            min_vals = self.min_vals.copy()
            min_vals.data = _min_vals
            max_vals = self.max_vals.copy()
            max_vals.data = _max_vals

            data_subjects = self.data_subjects

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
            if self.data_subjects != other.data_subjects:
                print("Entities are not the same?!?!?!")
                return self.gamma * other.gamma

            data = self.child * other.child

            min_min = self.min_vals.data * other.min_vals.data
            min_max = self.min_vals.data * other.max_vals.data
            max_min = self.max_vals.data * other.min_vals.data
            max_max = self.max_vals.data * other.max_vals.data

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
                elif isinstance(other, PhiTensor):
                    if self.data_subjects != other.data_subjects:
                        # return convert_to_gamma_tensor(self).__matmul__(convert_to_gamma_tensor(other))
                        raise NotImplementedError
                    else:
                        data = self.child.__matmul__(other.child)
                        # _min_vals = np.array(
                        #     [self.min_vals.data.__matmul__(other.min_vals.data)]
                        # )
                        # _max_vals = np.array(
                        #     [self.max_vals.data.__matmul__(other.max_vals.data)]
                        # )
                        # min_vals = self.min_vals.copy()
                        # min_vals.data = _min_vals
                        # max_vals = self.max_vals.copy()
                        # max_vals.data = _max_vals
                        min_vals = self.min_vals.__matmul__(other.min_vals)
                        max_vals = self.max_vals.__matmul__(other.max_vals)

                elif isinstance(other, GammaTensor):
                    return self.gamma @ other
                else:
                    print("Type is unsupported:" + str(type(other)))
                    raise NotImplementedError

                return PhiTensor(
                    child=data,
                    max_vals=max_vals,
                    min_vals=min_vals,
                    data_subjects=self.data_subjects,
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
                elif isinstance(other, PhiTensor):
                    if self.data_subjects != other.data_subjects:
                        # return convert_to_gamma_tensor(self).__matmul__(convert_to_gamma_tensor(other))
                        raise NotImplementedError
                    else:
                        data = self.child.__rmatmul__(other.child)
                        # _min_vals = np.array(
                        #     [self.min_vals.data.__matmul__(other.min_vals.data)]
                        # )
                        # _max_vals = np.array(
                        #     [self.max_vals.data.__matmul__(other.max_vals.data)]
                        # )
                        # min_vals = self.min_vals.copy()
                        # min_vals.data = _min_vals
                        # max_vals = self.max_vals.copy()
                        # max_vals.data = _max_vals
                        min_vals = self.min_vals.__rmatmul__(other.min_vals)
                        max_vals = self.max_vals.__rmatmul__(other.max_vals)

                else:
                    print("Type is unsupported:" + str(type(other)))
                    raise NotImplementedError

                return PhiTensor(
                    child=data,
                    max_vals=max_vals,
                    min_vals=min_vals,
                    data_subjects=self.data_subjects,
                )

    def transpose(self, *args: Any, **kwargs: Any) -> PhiTensor:
        """Transposes self.child, min_vals, and max_vals if these can be transposed, otherwise doesn't change them."""
        data: Sequence
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

        # TODO: Should we give warnings for min_val and max_val being single floats/integers/booleans too?
        if (
            isinstance(self.min_vals, int)
            or isinstance(self.min_vals, float)
            or isinstance(self.min_vals, bool)
        ):
            # For these data types, the transpose operation is meaningless, so don't change them.
            min_vals = self.min_vals
            # print(f'Warning: Tensor data was of type {type(data)}, transpose operation had no effect.')
        else:
            min_vals = self.min_vals.transpose(*args)

        if (
            isinstance(self.max_vals, int)
            or isinstance(self.max_vals, float)
            or isinstance(self.max_vals, bool)
        ):
            # For these data types, the transpose operation is meaningless, so don't change them.
            max_vals = self.max_vals
            # print(f'Warning: Tensor data was of type {type(data)}, transpose operation had no effect.')
        else:
            max_vals = self.max_vals.transpose(*args)

        return PhiTensor(
            child=data,
            data_subjects=self.data_subjects,
            min_vals=min_vals,
            max_vals=max_vals,
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

        # if the tensor being compared is also private
        if isinstance(other, PhiTensor):

            if self.data_subjects != other.data_subjects:
                # return self.gamma < other.gamma
                raise NotImplementedError

            if len(self.child) != len(other.child):
                raise Exception(
                    f"Tensor dims do not match for __lt__: {len(self.child)} != {len(other.child)}"  # type: ignore
                )

            data = (
                self.child < other.child
            )  # the * 1 just makes sure it returns integers instead of True/False
            min_vals = self.min_vals * 0
            max_vals = (self.max_vals * 0) + 1
            data_subjects = self.data_subjects

            return PhiTensor(
                child=data,
                data_subjects=data_subjects,
                min_vals=min_vals,
                max_vals=max_vals,
            )

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
            return NotImplementedError  # type: ignore

    def __gt__(self, other: SupportedChainType) -> Union[PhiTensor, GammaTensor]:

        # if the tensor being compared is also private
        if isinstance(other, PhiTensor):

            if self.data_subjects != other.data_subjects:
                # return self.gamma < other.gamma
                raise NotImplementedError

            if len(self.child) != len(other.child):
                raise Exception(
                    f"Tensor dims do not match for __gt__: {len(self.child)} != {len(other.child)}"  # type: ignore
                )

            data = (
                self.child > other.child
            )  # the * 1 just makes sure it returns integers instead of True/False
            min_vals = self.min_vals * 0
            max_vals = (self.max_vals * 0) + 1
            data_subjects = self.data_subjects

            return PhiTensor(
                child=data,
                data_subjects=data_subjects,
                min_vals=min_vals,
                max_vals=max_vals,
            )

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

    # Re enable after testing
    # def dot(
    #     self, other: Union[PhiTensor, GammaTensor, np.ndarray]
    # ) -> Union[PhiTensor, GammaTensor]:
    #     if isinstance(other, np.ndarray):
    #         print("We here or what?")
    #         return PhiTensor(
    #             child=np.dot(self.child, other),
    #             min_vals=np.dot(self.min_vals, other),
    #             max_vals=np.dot(self.max_vals, other),
    #             data_subjects=self.data_subjects,
    #         )
    #     elif isinstance(other, PhiTensor):
    #         if (
    #             len(self.data_subjects.one_hot_lookup) > 1
    #             or len(other.data_subjects.one_hot_lookup) > 1
    #         ):
    #             return self.gamma.dot(other.gamma)
    #         elif (
    #             len(self.data_subjects.one_hot_lookup) == 1
    #             and len(other.data_subjects.one_hot_lookup) == 1
    #             and self.data_subjects.one_hot_lookup != other.data_subjects.one_hot_lookup
    #         ):
    #             return self.gamma.dot(other.gamma)
    #     elif isinstance(other, GammaTensor):
    #         return self.gamma.dot(other)
    #     else:
    #         raise NotImplementedError

    def sum(
        self, axis: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> Union[PhiTensor, GammaTensor]:
        # TODO: Add support for axes arguments later
        min_val = lazyrepeatarray(data=np.array(self.min_vals.sum(axis=None)), shape=())
        max_val = lazyrepeatarray(data=np.array(self.max_vals.sum(axis=None)), shape=())
        if len(self.data_subjects.one_hot_lookup) == 1:
            return PhiTensor(
                child=self.child.sum(),
                min_vals=min_val,
                max_vals=max_val,
                data_subjects=DataSubjectList.from_objs(
                    self.data_subjects.one_hot_lookup[0]
                ),  # Need to check this
            )

        # TODO: Expand this later to include more args/kwargs
        res = GammaTensor(
            child=self.child.sum(),
            data_subjects=self.data_subjects.sum(),
            min_val=min_val,
            max_val=max_val,
        )
        print("Result", res)
        return res

    def __ne__(self, other: Any) -> Union[PhiTensor, GammaTensor]:  # type: ignore
        # TODO: what about data_subjects and min / max values?
        if is_acceptable_simple_type(other) or len(self.child) == len(other.child):
            gamma_output = False
            if is_acceptable_simple_type(other):
                result = self.child != other
            else:
                # check data_subjects match, if they dont gamma_output = True
                #
                result = self.child != other.child
                if isinstance(result, GammaTensor):
                    gamma_output = True
            if not gamma_output:
                return self.copy_with(child=result)
            else:
                return self.copy_with(child=result).gamma
        else:
            raise Exception(
                "Tensor dims do not match for __eq__: "
                + f"{len(self.child)} != {len(other.child)}"
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

    def _object2bytes(self) -> bytes:
        schema = get_capnp_schema(schema_file="phi_tensor.capnp")

        pt_struct: CapnpModule = schema.PT  # type: ignore
        pt_msg = pt_struct.new_message()
        # this is how we dispatch correct deserialization of bytes
        pt_msg.magicHeader = serde_magic_header(type(self))

        # We always have FPT as the child of an PT in the tensor chain.
        chunk_bytes(serialize(self.child, to_bytes=True), "child", pt_msg)  # type: ignore

        pt_msg.minVals = serialize(self.min_vals, to_bytes=True)
        pt_msg.maxVals = serialize(self.max_vals, to_bytes=True)
        pt_msg.dataSubjectsIndexed = capnp_serialize(
            self.data_subjects.data_subjects_indexed
        )

        pt_msg.oneHotLookup = capnp_serialize(
            liststrtonumpyutf8(self.data_subjects.one_hot_lookup)
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

        child = deserialize(combine_bytes(pt_msg.child), from_bytes=True)
        min_vals = deserialize(pt_msg.minVals, from_bytes=True)
        max_vals = deserialize(pt_msg.maxVals, from_bytes=True)
        data_subjects_indexed = capnp_deserialize(pt_msg.dataSubjectsIndexed)
        one_hot_lookup = numpyutf8tolist(capnp_deserialize(pt_msg.oneHotLookup))

        data_subjects_list = DataSubjectList(one_hot_lookup, data_subjects_indexed)

        return PhiTensor(
            child=child,
            min_vals=min_vals,
            max_vals=max_vals,
            data_subjects=data_subjects_list,
        )
