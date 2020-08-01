# external lib imports
import json

# external class/method imports
from google.protobuf import json_format

# syft import
from ....util import index_syft_by_module_name
from ..lazy_structures import LazyDict

class Serializable(object):
    def __init__(self):
        assert self.protobuf_type is not None

    def _proto2object(self, proto):
        raise NotImplementedError

    def _object2proto(self):
        raise NotImplementedError

    def to_json(self) -> str:
        return self.serialize(to_json=True)

    def json(self) -> str:
        return self.serialize(to_json=True)

    def to_binary(self) -> bytes:
        return self.serialize(to_binary=True)

    def binary(self) -> bytes:
        return self.serialize(to_binary=True)

    def to_hex(self) -> str:
        return self.serialize(to_hex=True)

    def hex(self) -> str:
        return self.serialize(to_hex=True)

    def serialize(self, to_json=True, to_binary=False, to_hex=False):
        """Serialize the object according to the parameters."""

        if to_json or to_binary or to_hex:
            blob = json_format.MessageToJson(message=self._object2proto())
            if to_binary or to_hex:
                blob = bytes(blob, "utf-8")
                if to_hex:
                    blob = blob.hex()
            return blob
        else:
            return json_format.MessageToDict(message=self._object2proto())

def is_string_a_serializable_class_name(self_dict, fully_qualified_name:str):
    obj_type = index_syft_by_module_name(fully_qualified_name=fully_qualified_name)
    if issubclass(obj_type, Serializable):
        self_dict._dict[fully_qualified_name] = obj_type
    elif hasattr(obj_type, 'serializable_wrapper'):
        self_dict._dict[fully_qualified_name] = obj_type.serializable_wrapper
    else:
        print(f"{fully_qualified_name} is not serializable")

string2type = LazyDict(update_rule=is_string_a_serializable_class_name)

def deserialize(
    blob: (str, dict, bytes), from_json=True, from_binary=False, from_hex=False
) -> Serializable:

    global string2type

    if from_hex:
        from_binary = True
        blob = bytes.fromhex(blob)

    if from_binary:
        from_json = True
        blob = str(blob, "utf-8")

    if from_json:
        blob = json.loads(s=blob)

    obj_type = string2type[blob['objType']]

    # try:
    #     # lets try to lookup the type we are deserializing
    #     obj_type = string2type[blob["objType"]]
    #
    # # uh-oh! Looks like the type doesn't exist.
    # except KeyError as e:
    #
    #     # It might be because we just haven't seen it before - let's try refreshing
    #     # our list of supported types.
    #     string2type = refresh_string2type()
    #
    #     # now that we've refreshed our list of supported types, let's try looking
    #     # up our type again.
    #     try:
    #         obj_type = string2type[blob["objType"]]
    #
    #     # uh-oh - looks like we *really* don't support this type. Let's throw an
    #     # informative error.
    #     except KeyError as e:
    #
    #         raise KeyError(
    #             """You tried to deserialize an unsupported type. This can be caused by
    #             several reasons. Either you are actively writing Syft code and forgot
    #             to create one, or you are trying to deserialize an object which was
    #             serialized using a different version of Syft and the object you tried
    #             to deserialize is not supported in this version."""
    #         )

    protobuf_type = obj_type.protobuf_type

    proto_obj = json_format.ParseDict(js_dict=blob, message=protobuf_type())

    return obj_type._proto2object(proto_obj)

