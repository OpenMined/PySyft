# external lib imports
import json

# external class/method imports
from google.protobuf import json_format

# syft import
from ...util import get_subclasses


class Serializable(object):
    def __init__(self):
        assert self.proto_type is not None

    def proto2object(self, proto):
        raise NotImplementedError

    def object2proto(self):
        raise NotImplementedError

    def serialize(self, to_json=True, to_binary=False, to_hex=False):
        """Serialize the object according to the parameters."""

        if to_json or to_binary or to_hex:
            blob = json_format.MessageToJson(message=self.object2proto())
            if to_binary or to_hex:
                blob = bytes(blob, "utf-8")
                if to_hex:
                    blob = blob.hex()
            return blob
        else:
            return json_format.MessageToDict(message=self.object2proto())


def refresh_string2type():
    classes = get_subclasses(obj_type=Serializable)

    string2type = {}

    for klass in classes:
        string = klass.__module__ + "." + klass.__name__
        string2type[string] = klass

    return string2type


string2type = refresh_string2type()


def deserialize(
    blob: (str, dict, bytes), from_json=True, from_binary=False, from_hex=False
) -> Serializable:

    global string2type

    if from_hex:
        from_binary=True
        blob = bytes.fromhex(blob)

    if from_binary:
        from_json = True
        blob = str(blob, 'utf-8')

    if from_json:
        blob = json.loads(s=blob)

    try:
        # lets try to lookup the type we are deserializing
        obj_type = string2type[blob["objType"]]

    # uh-oh! Looks like the type doesn't exist.
    except KeyError as e:

        # It might be because we just haven't seen it before - let's try refreshing
        # our list of supported types.
        string2type = refresh_string2type()

        # now that we've refreshed our list of supported types, let's try looking
        # up our type again.
        try:
            obj_type = string2type[blob["objType"]]

        # uh-oh - looks like we *really* don't support this type. Let's throw an
        # informative error.
        except KeyError as e:

            raise KeyError(
                """You tried to deserialize an unsupported type. This can be caused by 
                several reasons. Either you are actively writing Syft code and forgot 
                to create one, or you are trying to deserialize an object which was 
                serialized using a different version of Syft and the object you tried
                to deserialize is not supported in this version."""
            )

    proto_type = obj_type.proto_type

    proto_obj = json_format.ParseDict(js_dict=blob, message=proto_type())

    return obj_type.proto2object(proto_obj)
