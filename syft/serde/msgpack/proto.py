"""
This file exists to translate python classes to Serde type constants defined in `proto.json` file
in https://github.com/OpenMined/proto.
The reason for this is to have stable constants used in Serde serialization protocol
and the definition file that can be used not only by PySyft but also in other languages.

https://github.com/OpenMined/proto (`pysyft_proto` module) is included as dependency in setup.py
exposes contents of `proto.json` file in `proto_info` variable.

IMPORTANT: New types added in Serde need to be also defined in `proto.json`.
"""

from syft_proto import proto_info
from syft.exceptions import InvalidProtocolFileError
from syft.exceptions import UndefinedProtocolTypeError
from syft.exceptions import UndefinedProtocolTypePropertyError

if proto_info is None:
    raise InvalidProtocolFileError("Failed to load syft protocol data")


class TypeInfo:
    """Convenience wrapper for type info defined in `proto_info`.
    Exposes type constants with error handling.
    """

    def __init__(self, name, obj):
        """Initializes type info for a given class identified by `name` with contents of
        `proto_info` for this class.
        """
        self.name = name
        self.obj = obj

    @property
    def code(self):
        """Returns `code` property (serialization constant) for class
        or throws an exception if it's not defined in `proto.json`."""
        if "code" in self.obj:
            return self.obj["code"]
        else:
            raise UndefinedProtocolTypePropertyError(f"code is not set for {self.name}")

    @property
    def forced_code(self):
        """Returns `forced_code` property (serialization constant) for class
        or throws an exception if it's not defined in `proto.json`."""
        if "forced_code" in self.obj:
            return self.obj["forced_code"]
        else:
            raise UndefinedProtocolTypePropertyError(f"forced_code is not set for {self.name}")


def fullname(cls):
    """Returns full name of a given *class* (not instance of class).
    Source:
    https://stackoverflow.com/questions/2020014/get-fully-qualified-class-name-of-an-object-in-python. # noqa: E501
    """
    module = cls.__module__
    if module is None or module == str.__module__:
        return cls.__name__  # Avoid reporting __builtin__
    else:
        return module + "." + cls.__name__


def proto_type_info(cls):
    """Returns `TypeInfo` instance for a given *class* identified by `cls` parameter.
    Throws an exception when such class does not exists in the `proto.json`.
    """
    type_name = fullname(cls)

    if type_name in proto_info["TYPES"]:
        return TypeInfo(name=type_name, obj=proto_info["TYPES"][type_name])
    elif cls.get_msgpack_code.__qualname__.startswith(cls.__name__):
        return TypeInfo(name=type_name, obj=cls.get_msgpack_code())
    else:
        raise UndefinedProtocolTypeError(
            f"{type_name} is not defined in the protocol file and it does not provide a code by"
            f" implementing 'get_msgpack_code'."
        )
