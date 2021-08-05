# stdlib
from typing import List

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType

# syft absolute
from syft import deserialize
from syft import serialize

# relative
from ....core.common.serde.serializable import bind_protobuf
from ....lib.python.dict import Dict
from ....proto.core.common.recursive_serde_pb2 import (
    RecursiveSerde as RecursiveSerde_PB,
)
from ....util import get_fully_qualified_name
from ....util import index_syft_by_module_name
from .serializable import Serializable


@bind_protobuf
class RecursiveSerde(Serializable):
    """If you subclass from this object and put that subclass in the syft classpath somehow, then
    you'll be able to serialize it without having to create a custom protobuf. Be careful with this
    though, because it's going to include all attributes by default (including private data if
    it's there)."""

    # put attr names here - set this to None to include all attrs (not recommended)
    __attr_allowlist__: List[str] = []
<<<<<<< HEAD
    __serde_overrides__: dict = {}
=======
>>>>>>> d6688c7d1a2dea7ca122cde60e0a14b3690aa678

    def _object2proto(self) -> RecursiveSerde_PB:

        # if __attr_allowlist__ then only include attrs from that list
<<<<<<< HEAD
        if self.__attr_allowlist__ is not None or self.__serde_overrides__ is not None:
            attrs = {}
            for attr_name in self.__attr_allowlist__:
                if hasattr(self, attr_name):

                    attr = getattr(self, attr_name)

                    if attr_name in self.__serde_overrides__.keys():
                        attr = self.__serde_overrides__[attr_name][0](attr)

                    attrs[attr_name] = attr
=======
        if self.__attr_allowlist__ is not None:
            attrs = {}
            for attr_name in self.__attr_allowlist__:
                if hasattr(self, attr_name):
                    attrs[attr_name] = getattr(self, attr_name)
>>>>>>> d6688c7d1a2dea7ca122cde60e0a14b3690aa678

        # else include all attrs
        else:
            attrs = self.__dict__  # type: ignore

        return RecursiveSerde_PB(
<<<<<<< HEAD
            obj_type=get_fully_qualified_name(self),
            data=serialize(Dict(attrs), to_bytes=True),
=======
            data=serialize(Dict(attrs), to_bytes=True),
            fully_qualified_name=get_fully_qualified_name(self),
>>>>>>> d6688c7d1a2dea7ca122cde60e0a14b3690aa678
        )

    @staticmethod
    def _proto2object(proto: RecursiveSerde_PB) -> "RecursiveSerde":

        attrs = dict(deserialize(proto.data, from_bytes=True))

<<<<<<< HEAD
        class_type = index_syft_by_module_name(proto.obj_type)
=======
        class_type = index_syft_by_module_name(proto.fully_qualified_name)
>>>>>>> d6688c7d1a2dea7ca122cde60e0a14b3690aa678

        obj = object.__new__(class_type)  # type: ignore

        for attr_name, attr_value in attrs.items():
<<<<<<< HEAD

            if attr_name in class_type.__serde_overrides__.keys():
                attr_value = class_type.__serde_overrides__[attr_name][1](attr_value)

=======
>>>>>>> d6688c7d1a2dea7ca122cde60e0a14b3690aa678
            setattr(obj, attr_name, attr_value)

        return obj

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return RecursiveSerde_PB
