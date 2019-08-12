from pysyft_proto import proto_info
from syft.exceptions import InvalidProtocolFileError
from syft.exceptions import UndefinedProtocolTypeError
from syft.exceptions import UndefinedProtocolTypePropertyError

if proto_info is None:
    raise InvalidProtocolFileError("Failed to load syft protocol data")


class TypeInfo():
    def __init__(self, name, obj):
        self.name = name
        self.obj = obj

    @property
    def code(self):
        if "code" in self.obj:
            return self.obj["code"]
        else:
            raise UndefinedProtocolTypePropertyError("code is not set for %s" % self.name)

    @property
    def forced_code(self):
        if "forced_code" in self.obj:
            return self.obj["forced_code"]
        else:
            raise UndefinedProtocolTypePropertyError("forced_code is not set for %s" % self.name)


# https://stackoverflow.com/questions/2020014/get-fully-qualified-class-name-of-an-object-in-python
def fullname(cls):
  module = cls.__module__
  if module is None or module == str.__module__:
    return cls.__name__  # Avoid reporting __builtin__
  else:
    return module + '.' + cls.__name__


def proto_type_info(cls):
    type_name = fullname(cls)
    if type_name in proto_info["TYPES"]:
        return TypeInfo(name=type_name, obj=proto_info["TYPES"][type_name])
    else:
        raise UndefinedProtocolTypeError("%s is not defined in the protocol file" % type_name)

