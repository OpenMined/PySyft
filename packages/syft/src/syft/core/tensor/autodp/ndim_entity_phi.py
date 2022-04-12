# future
from __future__ import annotations

# stdlib
from collections.abc import Sequence
import operator
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

# third party
from nacl.signing import VerifyKey
import numpy as np

# relative
from .... import lib
from ....ast.klass import pointerize_args_and_kwargs
from ....core.adp.data_subject_ledger import DataSubjectLedger
from ....core.adp.data_subject_list import DataSubjectList
from ....core.adp.data_subject_list import liststrtonumpyutf8
from ....core.adp.data_subject_list import numpyutf8tolist
from ....core.adp.entity import Entity
from ....lib.numpy.array import capnp_deserialize
from ....lib.numpy.array import capnp_serialize
from ....lib.python.util import upcast
from ....util import inherit_tags
from ...common.serde.capnp import CapnpModule
from ...common.serde.capnp import get_capnp_schema
from ...common.serde.capnp import serde_magic_header
from ...common.serde.deserialize import _deserialize as deserialize
from ...common.serde.serializable import serializable
from ...common.serde.serialize import _serialize as serialize
from ...common.uid import UID
from ...node.abstract.node import AbstractNodeClient
from ...node.common.action.run_class_method_action import RunClassMethodAction
from ...pointer.pointer import Pointer
from ..ancestors import AutogradTensorAncestor
from ..config import DEFAULT_INT_NUMPY_TYPE
from ..fixed_precision_tensor import FixedPrecisionTensor
from ..lazy_repeat_array import lazyrepeatarray
from ..passthrough import AcceptableSimpleType  # type: ignore
from ..passthrough import PassthroughTensor  # type: ignore
from ..passthrough import SupportedChainType  # type: ignore
from ..passthrough import is_acceptable_simple_type  # type: ignore
from ..smpc import utils
from ..smpc.mpc_tensor import MPCTensor
from ..smpc.utils import TYPE_TO_RING_SIZE
from .adp_tensor import ADPTensor
from .gamma_tensor import GammaTensor
from .initial_gamma import InitialGammaTensor
from .initial_gamma import IntermediateGammaTensor


@serializable(recursive_serde=True)
class TensorWrappedNDimEntityPhiTensorPointer(Pointer):
    __name__ = "TensorWrappedNDimEntityPhiTensorPointer"
    __module__ = "syft.core.tensor.autodp.ndim_entity_phi"
    __attr_allowlist__ = [
        # default pointer attrs
        "client",
        "id_at_location",
        "object_type",
        "tags",
        "description",
        # ndim attrs
        "entities",
        "min_vals",
        "max_vals",
        "public_dtype",
        "public_shape",
    ]

    __serde_overrides__ = {
        "client": [lambda x: x.address, lambda y: y],
        "public_shape": [lambda x: x, lambda y: upcast(y)],
    }

    def __init__(
        self,
        entities: DataSubjectList,
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
        self.entities = entities
        self.public_shape = public_shape
        self.public_dtype = public_dtype

    # TODO: Modify for large arrays
    @property
    def synthetic(self) -> np.ndarray:
        return (
            np.random.rand(*list(self.public_shape))  # type: ignore
            * (self.max_vals.to_numpy() - self.min_vals.to_numpy())
            + self.min_vals.to_numpy()
        ).astype(self.public_dtype)

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

        # We always maintain a Tensor hierarchy Tensor ---> NDEPT--> Actual Data
        attr_path_and_name = f"syft.core.tensor.tensor.Tensor.__{op_str}__"

        result = TensorWrappedNDimEntityPhiTensorPointer(
            entities=self.entities,
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

        if isinstance(other, TensorWrappedNDimEntityPhiTensorPointer):
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
                f"Invalid Type for TensorWrappedNDimEntityPhiTensorPointer:{type(other)}"
            )

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
        self: TensorWrappedNDimEntityPhiTensorPointer,
        other: Union[
            TensorWrappedNDimEntityPhiTensorPointer, MPCTensor, int, float, np.ndarray
        ],
        op_str: str,
    ) -> Union[MPCTensor, TensorWrappedNDimEntityPhiTensorPointer]:
        """Performs the operation based on op_str

        Args:
            other (Union[TensorWrappedNDimEntityPhiTensorPointer,MPCTensor,int,float,np.ndarray]): second operand.

        Returns:
            Tuple[MPCTensor,Union[MPCTensor,int,float,np.ndarray]] : Result of the operation
        """
        op = getattr(operator, op_str)

        if (
            isinstance(other, TensorWrappedNDimEntityPhiTensorPointer)
            and self.client != other.client
        ):

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
        self,
        other: Union[
            TensorWrappedNDimEntityPhiTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Union[TensorWrappedNDimEntityPhiTensorPointer, MPCTensor]:
        """Apply the "add" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedNDimEntityPhiTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedNDimEntityPhiTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedNDimEntityPhiTensorPointer._apply_op(self, other, "add")

    def __sub__(
        self,
        other: Union[
            TensorWrappedNDimEntityPhiTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Union[TensorWrappedNDimEntityPhiTensorPointer, MPCTensor]:
        """Apply the "sub" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedNDimEntityPhiTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedNDimEntityPhiTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedNDimEntityPhiTensorPointer._apply_op(self, other, "sub")

    def __mul__(
        self,
        other: Union[
            TensorWrappedNDimEntityPhiTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Union[TensorWrappedNDimEntityPhiTensorPointer, MPCTensor]:
        """Apply the "mul" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedNDimEntityPhiTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedNDimEntityPhiTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedNDimEntityPhiTensorPointer._apply_op(self, other, "mul")

    def __matmul__(
        self,
        other: Union[
            TensorWrappedNDimEntityPhiTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Union[TensorWrappedNDimEntityPhiTensorPointer, MPCTensor]:
        """Apply the "matmul" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedNDimEntityPhiTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedNDimEntityPhiTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedNDimEntityPhiTensorPointer._apply_op(self, other, "matmul")

    def __lt__(
        self,
        other: Union[
            TensorWrappedNDimEntityPhiTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Union[TensorWrappedNDimEntityPhiTensorPointer, MPCTensor]:
        """Apply the "lt" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedNDimEntityPhiTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedNDimEntityPhiTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedNDimEntityPhiTensorPointer._apply_op(self, other, "lt")

    def __gt__(
        self,
        other: Union[
            TensorWrappedNDimEntityPhiTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Union[TensorWrappedNDimEntityPhiTensorPointer, MPCTensor]:
        """Apply the "gt" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedNDimEntityPhiTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedNDimEntityPhiTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedNDimEntityPhiTensorPointer._apply_op(self, other, "gt")

    def __ge__(
        self,
        other: Union[
            TensorWrappedNDimEntityPhiTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Union[TensorWrappedNDimEntityPhiTensorPointer, MPCTensor]:
        """Apply the "ge" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedNDimEntityPhiTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedNDimEntityPhiTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedNDimEntityPhiTensorPointer._apply_op(self, other, "ge")

    def __le__(
        self,
        other: Union[
            TensorWrappedNDimEntityPhiTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Union[TensorWrappedNDimEntityPhiTensorPointer, MPCTensor]:
        """Apply the "le" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedNDimEntityPhiTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedNDimEntityPhiTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedNDimEntityPhiTensorPointer._apply_op(self, other, "le")

    def __eq__(  # type: ignore
        self,
        other: Union[
            TensorWrappedNDimEntityPhiTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Union[TensorWrappedNDimEntityPhiTensorPointer, MPCTensor]:
        """Apply the "eq" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedNDimEntityPhiTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedNDimEntityPhiTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedNDimEntityPhiTensorPointer._apply_op(self, other, "eq")

    def __ne__(  # type: ignore
        self,
        other: Union[
            TensorWrappedNDimEntityPhiTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Union[TensorWrappedNDimEntityPhiTensorPointer, MPCTensor]:
        """Apply the "ne" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedNDimEntityPhiTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedNDimEntityPhiTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedNDimEntityPhiTensorPointer._apply_op(self, other, "ne")

    def to_local_object_without_private_data_child(self) -> NDimEntityPhiTensor:
        """Convert this pointer into a partial version of the NDimEntityPhiTensor but without
        any of the private data therein."""
        # relative
        from ..tensor import Tensor

        public_shape = getattr(self, "public_shape", None)
        public_dtype = getattr(self, "public_dtype", None)
        return Tensor(
            child=NDimEntityPhiTensor(
                child=FixedPrecisionTensor(value=None),
                entities=self.entities,
                min_vals=self.min_vals,  # type: ignore
                max_vals=self.max_vals,  # type: ignore
            ),
            public_shape=public_shape,
            public_dtype=public_dtype,
        )


@serializable(capnp_bytes=True)
class NDimEntityPhiTensor(PassthroughTensor, AutogradTensorAncestor, ADPTensor):
    PointerClassOverride = TensorWrappedNDimEntityPhiTensorPointer
    # __attr_allowlist__ = ["child", "min_vals", "max_vals", "entities"]
    __slots__ = (
        "child",
        "min_vals",
        "max_vals",
        "entities",
    )

    def __init__(
        self,
        child: Sequence,
        entities: Union[List[Entity], DataSubjectList],
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

        if not isinstance(entities, DataSubjectList):
            entities = DataSubjectList.from_objs(entities)

        self.entities = entities

    @property
    def proxy_public_kwargs(self) -> Dict[str, Any]:
        return {
            "min_vals": self.min_vals,
            "max_vals": self.max_vals,
            "entities": self.entities,
        }

    @staticmethod
    def from_rows(rows: Sequence) -> NDimEntityPhiTensor:
        # relative
        from .single_entity_phi import SingleEntityPhiTensor

        if len(rows) < 1 or not isinstance(rows[0], SingleEntityPhiTensor):
            raise Exception(
                "NDimEntityPhiTensor.from_rows requires a list of SingleEntityPhiTensors"
            )

        # create lazyrepeatarrays of the first element
        first_row = rows[0]
        min_vals = lazyrepeatarray(
            data=first_row.min_vals,
            shape=tuple([len(rows)] + list(first_row.min_vals.shape)),
        )
        max_vals = lazyrepeatarray(
            data=first_row.max_vals,
            shape=tuple([len(rows)] + list(first_row.max_vals.shape)),
        )

        # collect entities and children into numpy arrays
        entity_list = []
        child_list = []
        for row in rows:
            entity_list.append(row.entity)
            child_list.append(row.child)
        entities = DataSubjectList.from_objs(entities=entity_list)
        child = np.stack(child_list)

        # use new constructor
        return NDimEntityPhiTensor(
            child=child,
            min_vals=min_vals,
            max_vals=max_vals,
            entities=entities,
        )

    # def init_pointer(
    #     self,
    #     client: Any,
    #     id_at_location: Optional[UID] = None,
    #     object_type: str = "",
    #     tags: Optional[List[str]] = None,
    #     description: str = "",
    # ) -> TensorWrappedNDimEntityPhiTensorPointer:
    #     return TensorWrappedNDimEntityPhiTensorPointer(
    #         # Arguments specifically for SEPhiTensor
    #         entities=self.entities,
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

    def copy(self, order: Optional[str] = "K") -> NDimEntityPhiTensor:
        """Return copy of the given object"""

        return NDimEntityPhiTensor(
            child=self.child.copy(order=order),
            min_vals=self.min_vals.copy(order=order),
            max_vals=self.max_vals.copy(order=order),
            entities=self.entities.copy(order=order),
        )

    def all(self) -> bool:
        return self.child.all()

    def any(self) -> bool:
        return self.child.any()

    def copy_with(self, child: np.ndarray) -> NDimEntityPhiTensor:
        new_tensor = self.copy()
        new_tensor.child = child
        return new_tensor

    def create_gamma(self) -> GammaTensor:
        """Return a new Gamma tensor based on this phi tensor"""

        # if scalar_manager is None:
        #     scalar_manager = self.scalar_manager

        # Gamma expects an entity for each scalar
        # entities = np.array([self.entity] * np.array(self.child.shape).prod()).reshape(
        #     self.shape
        # )

        # TODO: update InitialGammaTensor to handle DataSubjectList
        # TODO: check if values needs to be a JAX array or if numpy will suffice

        return GammaTensor(
            value=self.child,
            data_subjects=self.entities,
            min_val=self.min_vals,
            max_val=self.max_vals,
        )

    def publish(
        self,
        get_budget_for_user: Callable,
        deduct_epsilon_for_user: Callable,
        ledger: DataSubjectLedger,
        sigma: float,
        user_key: VerifyKey,
    ) -> AcceptableSimpleType:
        print("PUBLISHING TO GAMMA:")
        print(self.child)
        return self.gamma.publish(
            get_budget_for_user=get_budget_for_user,
            deduct_epsilon_for_user=deduct_epsilon_for_user,
            ledger=ledger,
            sigma=sigma,
        )

    @property
    def value(self) -> np.ndarray:
        return self.child

    def astype(self, np_type: np.dtype) -> NDimEntityPhiTensor:
        return self.__class__(
            child=self.child.astype(np_type),
            entities=self.entities,
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

    def __eq__(  # type: ignore
        self, other: Any
    ) -> Union[NDimEntityPhiTensor, IntermediateGammaTensor, GammaTensor]:
        # TODO: what about entities and min / max values?
        if is_acceptable_simple_type(other) or len(self.child) == len(other.child):
            gamma_output = False
            if is_acceptable_simple_type(other):
                result = self.child == other
            else:
                # check entities match, if they dont gamma_output = True
                #
                result = self.child == other.child
                if isinstance(result, InitialGammaTensor):
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

    def __add__(
        self, other: SupportedChainType
    ) -> Union[NDimEntityPhiTensor, IntermediateGammaTensor, GammaTensor]:

        # if the tensor being added is also private
        if isinstance(other, NDimEntityPhiTensor):
            if self.entities != other.entities:
                return self.gamma + other.gamma

            return NDimEntityPhiTensor(
                child=self.child + other.child,
                min_vals=self.min_vals + other.min_vals,
                max_vals=self.max_vals + other.max_vals,
                entities=self.entities,
                # scalar_manager=self.scalar_manager,
            )

        # if the tensor being added is a public tensor / int / float / etc.
        elif is_acceptable_simple_type(other):
            return NDimEntityPhiTensor(
                child=self.child + other,
                min_vals=self.min_vals + other,
                max_vals=self.max_vals + other,
                entities=self.entities,
                # scalar_manager=self.scalar_manager,
            )

        elif isinstance(other, IntermediateGammaTensor):
            return self.gamma + other
        else:
            print("Type is unsupported:" + str(type(other)))
            raise NotImplementedError

    def sum(
        self, axis: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> Union[NDimEntityPhiTensor, GammaTensor]:
        # TODO: Add support for axes arguments later
        if len(self.entities.one_hot_lookup) == 1:
            return NDimEntityPhiTensor(
                child=self.child.sum(),
                min_vals=self.min_vals.sum(axis=None),
                max_vals=self.max_vals.sum(axis=None),
                entities=DataSubjectList.from_objs(
                    self.entities.one_hot_lookup[0]
                ),  # Need to check this
            )

        return GammaTensor(
            value=np.array(self.child.sum()),
            data_subjects=self.entities.sum(),
            min_val=float(self.min_vals.sum(axis=None)),
            max_val=float(self.max_vals.sum(axis=None)),
            inputs=self.child,
        )

    def __ne__(  # type: ignore
        self, other: Any
    ) -> Union[NDimEntityPhiTensor, IntermediateGammaTensor, GammaTensor]:
        # TODO: what about entities and min / max values?
        if is_acceptable_simple_type(other) or len(self.child) == len(other.child):
            gamma_output = False
            if is_acceptable_simple_type(other):
                result = self.child != other
            else:
                # check entities match, if they dont gamma_output = True
                #
                result = self.child != other.child
                if isinstance(result, InitialGammaTensor):
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

    def __neg__(self) -> NDimEntityPhiTensor:

        return NDimEntityPhiTensor(
            child=self.child * -1,
            min_vals=self.max_vals * -1,
            max_vals=self.min_vals * -1,
            entities=self.entities,
        )

    def _object2bytes(self) -> bytes:
        schema = get_capnp_schema(schema_file="ndept.capnp")

        ndept_struct: CapnpModule = schema.NDEPT  # type: ignore
        ndept_msg = ndept_struct.new_message()
        # this is how we dispatch correct deserialization of bytes
        ndept_msg.magicHeader = serde_magic_header(type(self))

        # TODO: move numpy serialization to capnp and modify child serialization
        # specificall done here for FPT
        ndept_msg.child = serialize(self.child, to_bytes=True)
        ndept_msg.minVals = serialize(self.min_vals, to_bytes=True)
        ndept_msg.maxVals = serialize(self.max_vals, to_bytes=True)
        ndept_msg.dataSubjectsIndexed = capnp_serialize(
            self.entities.data_subjects_indexed
        )

        ndept_msg.oneHotLookup = capnp_serialize(
            liststrtonumpyutf8(self.entities.one_hot_lookup)
        )

        # to pack or not to pack?
        # to_bytes = ndept_msg.to_bytes()

        return ndept_msg.to_bytes_packed()

    @staticmethod
    def _bytes2object(buf: bytes) -> NDimEntityPhiTensor:
        schema = get_capnp_schema(schema_file="ndept.capnp")
        ndept_struct: CapnpModule = schema.NDEPT  # type: ignore
        # https://stackoverflow.com/questions/48458839/capnproto-maximum-filesize
        MAX_TRAVERSAL_LIMIT = 2**64 - 1
        # to pack or not to pack?
        # ndept_msg = ndept_struct.from_bytes(buf, traversal_limit_in_words=2 ** 64 - 1)
        ndept_msg = ndept_struct.from_bytes_packed(
            buf, traversal_limit_in_words=MAX_TRAVERSAL_LIMIT
        )

        child = deserialize(ndept_msg.child, from_bytes=True)
        min_vals = deserialize(ndept_msg.minVals, from_bytes=True)
        max_vals = deserialize(ndept_msg.maxVals, from_bytes=True)
        data_subjects_indexed = capnp_deserialize(ndept_msg.dataSubjectsIndexed)
        one_hot_lookup = numpyutf8tolist(capnp_deserialize(ndept_msg.oneHotLookup))

        entity_list = DataSubjectList(one_hot_lookup, data_subjects_indexed)

        return NDimEntityPhiTensor(
            child=child, min_vals=min_vals, max_vals=max_vals, entities=entity_list
        )
