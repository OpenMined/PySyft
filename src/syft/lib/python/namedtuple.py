# stdlib
from collections import namedtuple
from typing import Any
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

# third party
from google.protobuf.message import Message
from google.protobuf.reflection import GeneratedProtocolMessageType
import torch

# syft relative
from ...core.common.serde.deserialize import _deserialize
from ...core.common.serde.serializable import Serializable
from ...core.common.serde.serialize import _serialize
from ...core.common.uid import UID
from ...core.store.storeable_object import StorableObject
from ...lib.util import full_name_with_qualname
from ...proto.lib.torch.valuesindices_pb2 import ValuesIndicesProto as ValuesIndices_PB
from ...util import aggressive_set_attr
from ..torch.tensor_util import protobuf_tensor_deserializer
from ..torch.tensor_util import protobuf_tensor_serializer

ValuesIndices = namedtuple("ValuesIndices", "values indices")


class ValuesIndicesWrapper(StorableObject):
    def __init__(self, value: object):
        _id = getattr(value, "id", UID())
        obj_type, values, indices = ValuesIndicesWrapper.get_parts(return_tuple=value)
        return_tuple = ValuesIndicesWrapper.make_namedtuple(
            obj_type=obj_type, values=values, indices=indices, id=_id
        )

        super().__init__(
            data=return_tuple,
            id=_id,
            tags=getattr(value, "tags", []),
            description=getattr(value, "description", ""),
        )

        self.value = return_tuple

    def _data_object2proto(self) -> ValuesIndices_PB:
        values = getattr(self.data, "values", None)
        indices = getattr(self.data, "indices", None)

        proto = ValuesIndices_PB()
        proto.id.CopyFrom(_serialize(obj=self.id))
        proto.obj_type = full_name_with_qualname(klass=type(self.data))
        proto.values.CopyFrom(protobuf_tensor_serializer(values))
        proto.indices.CopyFrom(protobuf_tensor_serializer(indices))

        return proto

    @staticmethod
    def _data_proto2object(proto: ValuesIndices_PB) -> "ValuesIndices":  # type: ignore
        _id: UID = _deserialize(blob=proto.id)
        values = protobuf_tensor_deserializer(proto.values)
        indices = protobuf_tensor_deserializer(proto.indices)

        return_type = ValuesIndicesWrapper.make_namedtuple(
            obj_type=proto.obj_type, values=values, indices=indices, id=_id
        )

        return return_type

    @staticmethod
    def get_parts(return_tuple: Any) -> Tuple[torch.Tensor, torch.Tensor, str]:
        obj_type = full_name_with_qualname(klass=type(return_tuple))
        values = return_tuple.values
        indices = return_tuple.indices

        return (obj_type, values, indices)

    @staticmethod
    def make_namedtuple(
        obj_type: str,
        values: torch.Tensor,
        indices: torch.Tensor,
        id: UID,
        tags: List[str] = [],
        description: str = "",
    ) -> Any:
        module_parts = obj_type.split(".")
        klass = module_parts.pop()
        module_name = ".".join(module_parts)
        tuple_klass = namedtuple(  # type: ignore
            klass, ("values", "indices", "tags", "description", "id")
        )
        tuple_klass.__module__ = module_name
        return tuple_klass(  # type: ignore
            values=values, indices=indices, tags=tags, description=description, id=id
        )

    @staticmethod
    def get_data_protobuf_schema() -> GeneratedProtocolMessageType:
        return ValuesIndices_PB

    @staticmethod
    def construct_new_object(
        id: UID,
        data: StorableObject,
        description: Optional[str],
        tags: Optional[List[str]],
    ) -> StorableObject:
        obj_type, values, indices = ValuesIndicesWrapper.get_parts(return_tuple=data)

        if tags is None:
            # for the type checker
            tags = []
        if description is None:
            # for the type checker
            description = ""

        return_tuple = ValuesIndicesWrapper.make_namedtuple(
            obj_type=obj_type,
            values=values,
            indices=indices,
            id=id,
            tags=tags,
            description=description,
        )

        return return_tuple


# get each of the dynamic torch.return_types.*
def add_torch_return_types() -> None:
    x = torch.Tensor([1, 2, 3])
    y = x.cummax(0)

    supported_types = []
    supported_types.append(type(y))

    for types in supported_types:
        aggressive_set_attr(
            obj=types, name="serializable_wrapper_type", attr=ValuesIndicesWrapper
        )

        def attr_serialize(  # type: ignore
            self,
            to_proto: bool = True,
            to_bytes: bool = False,
        ) -> Union[str, bytes, Message]:
            return _serialize(
                obj=self,
                to_proto=to_proto,
                to_bytes=to_bytes,
            )

        aggressive_set_attr(obj=types, name="serialize", attr=attr_serialize)
        aggressive_set_attr(obj=types, name="to_proto", attr=Serializable.to_proto)
        aggressive_set_attr(obj=types, name="proto", attr=Serializable.proto)
        aggressive_set_attr(obj=types, name="to_bytes", attr=Serializable.to_bytes)


add_torch_return_types()
