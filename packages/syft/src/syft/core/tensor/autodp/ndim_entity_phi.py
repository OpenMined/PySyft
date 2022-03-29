# future
from __future__ import annotations

# stdlib
from collections.abc import Sequence
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

# third party
from nacl.signing import VerifyKey
import numpy as np

# relative
from ....core.adp.entity import Entity
from ....core.adp.entity_list import EntityList
from ....lib.numpy.array import arrow_deserialize as numpy_deserialize
from ....lib.numpy.array import arrow_serialize as numpy_serialize
from ...adp.vm_private_scalar_manager import VirtualMachinePrivateScalarManager
from ...common.serde.capnp import CapnpModule
from ...common.serde.capnp import chunk_bytes
from ...common.serde.capnp import combine_bytes
from ...common.serde.capnp import get_capnp_schema
from ...common.serde.capnp import serde_magic_header
from ...common.serde.serializable import serializable
from ...common.uid import UID
from ...pointer.pointer import Pointer
from ..ancestors import AutogradTensorAncestor
from ..lazy_repeat_array import lazyrepeatarray
from ..passthrough import AcceptableSimpleType  # type: ignore
from ..passthrough import PassthroughTensor  # type: ignore
from ..passthrough import SupportedChainType  # type: ignore
from ..passthrough import is_acceptable_simple_type  # type: ignore
from .adp_tensor import ADPTensor
from .initial_gamma import InitialGammaTensor
from .initial_gamma import IntermediateGammaTensor
from .single_entity_phi import SingleEntityPhiTensor


@serializable(recursive_serde=True)
class NDimEntityPhiTensorPointer(Pointer):
    __name__ = "NDimEntityPhiTensorPointer"
    __module__ = "syft.core.tensor.autodp.ndim_entity_phi"
    __attr_allowlist__ = [
        # default pointer attrs
        "points_to_object_with_path",
        "pointer_name",
        "id_at_location",
        "location",
        "tags",
        "description",
        "object_type",
        "attribute_name",
        "public_shape",
        "_exhausted",
        "gc_enabled",
        "is_enum",
        # ndim attrs
        "entities",
        "min_vals",
        "max_vals",
        "client",
        "public_dtype",
    ]

    # TODO :should create serialization for Entity List

    def __init__(
        self,
        entities: EntityList,
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


@serializable(capnp_bytes=True)
class NDimEntityPhiTensor(PassthroughTensor, AutogradTensorAncestor, ADPTensor):
    PointerClassOverride = NDimEntityPhiTensorPointer
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
        entities: Union[List[Entity], EntityList],
        min_vals: np.ndarray,
        max_vals: np.ndarray,
        row_type: SingleEntityPhiTensor = SingleEntityPhiTensor,  # type: ignore
    ) -> None:

        # child = the actual private data
        super().__init__(child)

        # lazyrepeatarray matching the shape of child
        if not isinstance(min_vals, lazyrepeatarray):
            min_vals = lazyrepeatarray(data=min_vals, shape=child.shape)  # type: ignore
        if not isinstance(max_vals, lazyrepeatarray):
            max_vals = lazyrepeatarray(data=max_vals, shape=child.shape)  # type: ignore
        self.min_vals = min_vals
        self.max_vals = max_vals

        if not isinstance(entities, EntityList):
            entities = EntityList.from_objs(entities)

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
        entities = EntityList.from_objs(entities=entity_list)
        child = np.stack(child_list)

        # use new constructor
        return NDimEntityPhiTensor(
            child=child,
            min_vals=min_vals,
            max_vals=max_vals,
            entities=entities,
        )

    def init_pointer(
        self,
        client: Any,
        id_at_location: Optional[UID] = None,
        object_type: str = "",
        tags: Optional[List[str]] = None,
        description: str = "",
    ) -> NDimEntityPhiTensorPointer:
        return NDimEntityPhiTensorPointer(
            # Arguments specifically for SEPhiTensor
            entities=self.entities,
            min_vals=self.min_vals,
            max_vals=self.max_vals,
            # Arguments required for a Pointer to work
            client=client,
            id_at_location=id_at_location,
            object_type=object_type,
            tags=tags,
            description=description,
        )

    @property
    def gamma(self) -> InitialGammaTensor:
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

    def create_gamma(
        self, scalar_manager: Optional[VirtualMachinePrivateScalarManager] = None
    ) -> InitialGammaTensor:
        """Return a new Gamma tensor based on this phi tensor"""

        # if scalar_manager is None:
        #     scalar_manager = self.scalar_manager

        # Gamma expects an entity for each scalar
        # entities = np.array([self.entity] * np.array(self.child.shape).prod()).reshape(
        #     self.shape
        # )

        # TODO: update InitialGammaTensor to handle EntityList
        return InitialGammaTensor(
            values=self.child,
            min_vals=self.min_vals,
            max_vals=self.max_vals,
            entities=self.entities,
            # scalar_manager=scalar_manager,
        )

    def publish(
        self, acc: Any, sigma: float, user_key: VerifyKey
    ) -> AcceptableSimpleType:
        print("PUBLISHING TO GAMMA:")
        print(self.child)
        return self.gamma.publish(acc=acc, sigma=sigma, user_key=user_key)

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
            f"{self.__class__.__name__}(child={self.child.shape}, "
            + f"min_vals={self.min_vals}, max_vals={self.max_vals})"
        )

    def __eq__(self, other: Any) -> Union[NDimEntityPhiTensor, IntermediateGammaTensor]:
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
    ) -> Union[NDimEntityPhiTensor, IntermediateGammaTensor]:

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

    def _object2bytes(self) -> bytes:
        schema = get_capnp_schema(schema_file="ndept.capnp")

        rows, rows_size = numpy_serialize(self.child, get_bytes=True)
        min_vals, min_vals_size = numpy_serialize(self.min_vals.data, get_bytes=True)
        max_vals, max_vals_size = numpy_serialize(self.max_vals.data, get_bytes=True)
        entities_indexed, entities_indexed_size = numpy_serialize(
            self.entities.entities_indexed, get_bytes=True
        )
        one_hot_lookup = self.entities.one_hot_lookup

        ndept_struct: CapnpModule = schema.NDEPT  # type: ignore
        ndept_msg = ndept_struct.new_message()
        metadata_schema = ndept_struct.TensorMetadata
        child_metadata = metadata_schema.new_message()
        min_vals_metadata = metadata_schema.new_message()
        max_vals_metadata = metadata_schema.new_message()
        entities_metadata = metadata_schema.new_message()

        # this is how we dispatch correct deserialization of bytes
        ndept_msg.magicHeader = serde_magic_header(type(self))

        chunk_bytes(rows, "child", ndept_msg)
        child_metadata.dtype = str(self.child.dtype)
        child_metadata.decompressedSize = rows_size
        ndept_msg.childMetadata = child_metadata

        chunk_bytes(min_vals, "minVals", ndept_msg)
        min_vals_metadata.dtype = str(self.min_vals.data.dtype)
        min_vals_metadata.decompressedSize = min_vals_size
        ndept_msg.minValsMetadata = min_vals_metadata

        chunk_bytes(max_vals, "maxVals", ndept_msg)
        max_vals_metadata.dtype = str(self.max_vals.data.dtype)
        max_vals_metadata.decompressedSize = max_vals_size
        ndept_msg.maxValsMetadata = max_vals_metadata

        chunk_bytes(entities_indexed, "entitiesIndexed", ndept_msg)
        entities_metadata.dtype = str(self.entities.entities_indexed.dtype)
        entities_metadata.decompressedSize = entities_indexed_size
        ndept_msg.entitiesIndexedMetadata = entities_metadata

        oneHotLookupList = ndept_msg.init("oneHotLookup", len(one_hot_lookup))
        for i, entity in enumerate(one_hot_lookup):
            oneHotLookupList[i] = (
                entity if not getattr(entity, "name", None) else entity.name  # type: ignore
            )

        # to pack or not to pack?
        # return ndept_msg.to_bytes()
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

        child_metadata = ndept_msg.childMetadata

        child = numpy_deserialize(
            combine_bytes(ndept_msg.child),
            child_metadata.decompressedSize,
            child_metadata.dtype,
        )

        min_vals_metadata = ndept_msg.minValsMetadata
        min_vals = lazyrepeatarray(
            numpy_deserialize(
                combine_bytes(ndept_msg.minVals),
                min_vals_metadata.decompressedSize,
                min_vals_metadata.dtype,
            ),
            child.shape,
        )

        max_vals_metadata = ndept_msg.maxValsMetadata
        max_vals = lazyrepeatarray(
            numpy_deserialize(
                combine_bytes(ndept_msg.maxVals),
                max_vals_metadata.decompressedSize,
                max_vals_metadata.dtype,
            ),
            child.shape,
        )

        entities_metadata = ndept_msg.entitiesIndexedMetadata
        entities_indexed = numpy_deserialize(
            combine_bytes(ndept_msg.entitiesIndexed),
            entities_metadata.decompressedSize,
            entities_metadata.dtype,
        )
        one_hot_lookup = np.array(ndept_msg.oneHotLookup)

        entity_list = EntityList(one_hot_lookup, entities_indexed)

        return NDimEntityPhiTensor(
            child=child, min_vals=min_vals, max_vals=max_vals, entities=entity_list
        )
