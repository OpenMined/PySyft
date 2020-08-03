# external lib imports
import json

# external class/method imports
from google.protobuf import json_format

from syft.core.common.lazy_structures import LazyDict

# syft import
from syft.util import get_fully_qualified_name, index_syft_by_module_name


class Serializable(object):
    def __init__(self, as_wrapper: bool):
        assert self.protobuf_type is not None
        self.as_wrapper = as_wrapper

    @staticmethod
    def _proto2object(proto):
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


def _is_string_a_serializable_class_name(lazy_dict, fully_qualified_name: str):

    """This method exists to allow a LazyDict to determine whether an
    object should actually be in its store - aka has the LazyDict been
    lazy and forgotten to add this object thus far.

    In particular, this is the method for the LazyDict within the fully_qualified_name2type
    dictionary - which is used to map fully qualified module names,
    (i.e., 'syft.core.common.UID') to their type object.

    So this method is designed to ask the question, 'Is self_dict an object
    we can serialize?' If it is, we add it to the LazyDict by adding it to
    lazy_dict._dict. If not, we do nothing.

    We determine whether we can serialize the object according to series of
    checks - as outlined below."""

    # lookup the type from the fully qualified name
    # i.e. "syft.core.common.UID" -> <type UID>
    obj_type = index_syft_by_module_name(fully_qualified_name=fully_qualified_name)

    # Check 1: If the object is a subclass of Serializable, then we can serialize it
    # add it to the lazy dictionary.
    if issubclass(obj_type, Serializable):
        lazy_dict._dict[fully_qualified_name] = obj_type

    # Check 2: If the object a non-syft object which is wrapped by a serializable
    # syft object? Aka, since we can't make non-syft objects subclass from
    # Serializable, have we created a wrapper around this object which does
    # subclass from serializable. Note that we can find out by seeing if we
    # monkeypatched a .serializable_wrapper attribute onto this non-syft class.
    elif hasattr(obj_type, "serializable_wrapper_type"):

        # this 'wrapper' object is a syft object which subclasses Serializable
        # so that we can put logic into it showing how to serialize and
        # deserialize the non-syft object.
        wrapper_type = obj_type.serializable_wrapper_type

        # just renaming the variable since we know something about this variable now
        # just so the code reads easier (the compile will remove this so it won't
        # affect performance)
        non_syft_object_fully_qualified_name = fully_qualified_name
        wrapper_type_fully_qualified_name = get_fully_qualified_name(wrapper_type)

        # so now we should update the dictionary so that in the future we can
        # quickly find the wrapper type from both the non_syft_object's fully
        # qualified name and the wrapper_type's fully qualified name
        lazy_dict[wrapper_type_fully_qualified_name] = wrapper_type
        lazy_dict[non_syft_object_fully_qualified_name] = wrapper_type

    else:
        raise Exception(f"{fully_qualified_name} is not serializable")


fully_qualified_name2type = LazyDict(update_rule=_is_string_a_serializable_class_name)


def _serialize(
    obj: (Serializable, object), to_json=True, to_binary=False, to_hex=False
):

    if not isinstance(obj, Serializable):
        obj = obj.serializable_wrapper_type(value=obj, as_wrapper=True)

    return obj.serialize(to_json=to_json, to_binary=to_binary, to_hex=to_hex)


def _deserialize(
    blob: (str, dict, bytes), from_json=True, from_binary=False, from_hex=False
) -> Serializable:

    global fully_qualified_name2type

    if from_hex:
        from_binary = True
        blob = bytes.fromhex(blob)

    if from_binary:
        from_json = True
        blob = str(blob, "utf-8")

    if from_json:
        blob = json.loads(s=blob)

    try:
        # lets try to lookup the type we are deserializing
        obj_type = fully_qualified_name2type[blob["objType"]]

    # uh-oh! Looks like the type doesn't exist. Let's throw an informative error.
    except KeyError:

        raise KeyError(
            """You tried to deserialize an unsupported type. This can be caused by
            several reasons. Either you are actively writing Syft code and forgot
            to create one, or you are trying to deserialize an object which was
            serialized using a different version of Syft and the object you tried
            to deserialize is not supported in this version."""
        )

    protobuf_type = obj_type.protobuf_type

    proto_obj = json_format.ParseDict(js_dict=blob, message=protobuf_type())

    return obj_type._proto2object(proto_obj)
