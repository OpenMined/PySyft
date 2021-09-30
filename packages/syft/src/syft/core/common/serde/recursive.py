# stdlib
from typing import Any

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType

# syft absolute
import syft as sy

# relative
from ....proto.core.common.recursive_serde_pb2 import (
    RecursiveSerde as RecursiveSerde_PB,
)
from ....util import get_fully_qualified_name
from ....util import index_syft_by_module_name


def rs_object2proto(self: Any) -> RecursiveSerde_PB:
    # if __attr_allowlist__ then only include attrs from that list
    msg = RecursiveSerde_PB(fully_qualified_name=get_fully_qualified_name(self))

    if self.__attr_allowlist__ is None:
        attribute_dict = self.__dict__.keys()
    else:
        attribute_dict = self.__attr_allowlist__

    for attr_name in attribute_dict:
        if hasattr(self, attr_name):
            msg.fields_name.append(attr_name)
            transforms = self.__serde_overrides__.get(attr_name, None)
            if transforms is None:
                field_obj = getattr(self, attr_name)
            else:
                field_obj = transforms[0](getattr(self, attr_name))
            msg.fields_data.append(sy.serialize(field_obj, to_bytes=True))
    return msg


def rs_proto2object(proto: RecursiveSerde_PB) -> Any:
    class_type = index_syft_by_module_name(proto.fully_qualified_name)
    obj = class_type.__new__(class_type)  # type: ignore
    for attr_name, attr_bytes in zip(proto.fields_name, proto.fields_data):
        attr_value = sy.deserialize(attr_bytes, from_bytes=True)
        transforms = obj.__serde_overrides__.get(attr_name, None)
        try:
            if transforms is None:
                setattr(obj, attr_name, attr_value)
            else:
                setattr(obj, attr_name, transforms[1](attr_value))
        except AttributeError:
            # if its an ID we need to set the _id instead
            if attr_name == "id":
                attr_name = "_id"
                if transforms is None:
                    setattr(obj, attr_name, attr_value)
                else:
                    setattr(obj, attr_name, transforms[1](attr_value))

    return obj


def rs_get_protobuf_schema() -> GeneratedProtocolMessageType:
    return RecursiveSerde_PB
