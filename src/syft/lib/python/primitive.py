import json
from typing import Any
from typing import Optional
from typing import List

from google.protobuf.reflection import GeneratedProtocolMessageType

import syft as sy

from ...proto.lib.python.python_primitive_pb2 import (
    PythonPrimitive as PythonPrimitive_PB,
)

from ...core.common.serde import _deserialize
from ...decorators.syft_decorator_impl import syft_decorator
from ...core.store.storeable_object import StorableObject
from ...core.common.uid import UID


@syft_decorator(typechecking=True)
def isprimitive(value: Any) -> bool:
    # check primitive types first
    if type(value) in [type(None), bool, int, float]:
        return True

    # if the type is a collection lets recursively search
    # set cant be easily converted to json
    if type(value) in [tuple, list]:
        for sub in value:
            is_subprimitive = isprimitive(value=sub)
            # it has failed so we should reject
            if not is_subprimitive:
                return False
        return True

    if type(value) is dict:
        for k, sub in value.items():
            is_k = type(k) == str
            if not is_k:
                return False
            is_v = isprimitive(value=sub)
            if not is_v:
                return False
        return True
    # cant find type
    return False


class PythonPrimitive(StorableObject):

    value: Any

    def __init__(self, value: Any, id: Optional[UID] = None) -> None:
        super().__init__(
            data=value,
            id=id if id is not None else UID(),
            tags=getattr(value, "tags", []),
            description=getattr(value, "description", ""),
        )
        self.value = value

    @property
    def icon(self) -> str:
        return "🗿"

    @property
    def pprint(self) -> str:
        return f"{self.icon} ({type(self.value).__name__}) {self.value.__repr__()}"

    @property
    def is_valid(self) -> bool:
        return isprimitive(self.value)

    def _data_object2proto(self) -> PythonPrimitive_PB:
        if sy.VERBOSE:
            print(f"> {self.icon} -> Proto 🔢")

        return PythonPrimitive_PB(
            id=self.id.serialize(),
            obj_type=type(self.value).__name__,
            content=json.dumps(self.value),
        )

    @staticmethod
    def _data_proto2object(proto: PythonPrimitive_PB) -> Any:
        obj = PythonPrimitive(
            value=json.loads(proto.content), id=_deserialize(blob=proto.id),
        )

        if sy.VERBOSE:
            icon = "🤷🏾‍♀️"
            if hasattr(obj, "icon"):
                icon = obj.icon
            print(f"> {icon} <- 🔢 Proto")

        if type(obj.value).__name__ != proto.obj_type:
            raise TypeError(
                "Deserializing Python Primitive JsonMessage. "
                + f"Expected type {proto.obj_type}. Got {type(obj.value).__name__}"
            )

        return obj

    @staticmethod
    def get_data_protobuf_schema() -> GeneratedProtocolMessageType:
        return PythonPrimitive_PB

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return PythonPrimitive_PB

    @staticmethod
    def construct_new_object(
        id: UID,
        data: StorableObject,
        description: Optional[str],
        tags: Optional[List[str]],
    ) -> StorableObject:
        data.id = id
        data.tags = tags
        data.description = description
        return data
