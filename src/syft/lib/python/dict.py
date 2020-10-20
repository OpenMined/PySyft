# stdlib
from collections import UserDict
from typing import Any
from typing import List
from typing import Optional
from typing import Iterable
from typing import Mapping
import warnings

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType

# syft relative
from ... import deserialize
from ... import serialize
from ...core.common import UID
from ...core.store.storeable_object import StorableObject
from ...decorators import syft_decorator
from ...proto.lib.python.dict_pb2 import Dict as Dict_PB
from ...util import aggressive_set_attr
from .primitive_interface import PyPrimitive
from .primitive_factory import PrimitiveFactory
from .util import downcast


class Dict(UserDict, PyPrimitive):
    # the incoming types to UserDict __init__ are overloaded and weird
    # see https://github.com/python/cpython/blob/master/Lib/collections/__init__.py
    # this is the version from python 3.7 because we need to support 3.6 and 3.7
    # python 3.8 signature includes a new PEP 570 (args, /, kwargs) syntax:
    # https://www.python.org/dev/peps/pep-0570/
    @syft_decorator(typechecking=False, prohibit_args=False)
    def __init__(*args: Any, **kwargs: Any) -> None:
        if not args:
            raise TypeError(
                "descriptor '__init__' of 'Dict' object " "needs an argument"
            )
        self, *args = args  # type: ignore
        if len(args) > 1:
            raise TypeError("expected at most 1 arguments, got %d" % len(args))
        if args:
            args_dict = args[0]
        elif "dict" in kwargs:
            args_dict = kwargs.pop("dict")

            warnings.warn(
                "Passing 'dict' as keyword argument is deprecated",
                DeprecationWarning,
                stacklevel=2,
            )
        else:
            args_dict = None

        self.data = {}
        if args_dict is not None:
            self.update(args_dict)
        if kwargs:
            self.update(kwargs)

        # We cant add UID from kwargs or it could easily be overwritten by the dict
        # that is being passed in for __init__
        # If you want to update it use the _id setter after creation.
        self._id = UID()

    # fix the type signature
    __init__.__text_signature__ = "($self, dict=None, /, **kwargs)"

    @property
    def id(self) -> UID:
        """We reveal PyPrimitive.id as a property to discourage users and
        developers of Syft from modifying .id attributes after an object
        has been initialized.

        :return: returns the unique id of the object
        :rtype: UID
        """
        return self._id

    @syft_decorator(typechecking=True, prohibit_args=True)
    def upcast(self) -> dict:
        return dict(self)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __contains__(self, other: Any) -> PyPrimitive:
        res = super().__contains__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __eq__(self, other: Any) -> PyPrimitive:
        res = super().__eq__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __format__(self, format_spec: str) -> PyPrimitive:
        res = super().__format__(format_spec)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __ge__(self, other: Any) -> PyPrimitive:
        res = super().__ge__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __getitem__(self, key: Any) -> PyPrimitive:
        res = super().__getitem__(key)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __gt__(self, other: Any) -> PyPrimitive:
        res = super().__gt__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __hash__(self) -> PyPrimitive:
        res = super().__hash__()
        return PrimitiveFactory.generate_primitive(value=res)

    #TODO fix this
    # @syft_decorator(typechecking=True, prohibit_args=False)
    # def __iter__(self) -> PyPrimitive:
    #     res = super().__iter__()
    #     return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __le__(self, other: Any) -> PyPrimitive:
        res = super().__le__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __len__(self) -> PyPrimitive:
        res = super().__len__()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __lt__(self, other: Any) -> PyPrimitive:
        res = super().__lt__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __ne__(self, other: Any) -> PyPrimitive:
        res = super().__ne__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __sizeof__(self) -> PyPrimitive:
        res = super().__sizeof__()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __str__(self) -> PyPrimitive:
        res = super().__str__()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def copy(self) -> PyPrimitive:
        res = super().copy()
        return PrimitiveFactory.generate_primitive(value=res)

    @classmethod
    @syft_decorator(typechecking=True, prohibit_args=False)
    def fromkeys(cls, iterable: Iterable, value: Optional[Any] = ...) -> PyPrimitive:
        res = super().fromkeys(iterable, value)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def get(self, key: Any) -> PyPrimitive:
        res = super().get(key)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def items(self) -> PyPrimitive:
        res = super().items()
        # TODO: we should actually support this at some point and stop converting to list
        res = list(res)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def keys(self) -> PyPrimitive:
        res = super().keys()
        # TODO: we should actually support this at some point and stop converting to list
        res = list(res)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def pop(self, key: Any) -> PyPrimitive:
        res = super().pop(key)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def popitem(self) -> PyPrimitive:
        res = super().popitem()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def setdefault(self, key: Any = ..., default: Any = ...) -> PyPrimitive:
        res = super().setdefault(key, default)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def values(self) -> PyPrimitive:
        res = super().values()
        res = list(res)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def clear(self) -> PyPrimitive:
        res = super().clear()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> Dict_PB:
        id_ = serialize(obj=self.id)
        keys = [serialize(obj=element) for element in self.data.keys()]
        values = [serialize(obj=element) for element in self.data.values()]
        return Dict_PB(id=id_, keys=keys, values=values)

    @staticmethod
    @syft_decorator(typechecking=True)
    def _proto2object(proto: Dict_PB) -> "Dict":
        id_: UID = deserialize(blob=proto.id)
        values = [deserialize(blob=downcast(value=element)) for element in proto.values]
        keys = [deserialize(blob=downcast(value=element)) for element in proto.keys]
        new_dict = Dict(dict(zip(keys, values)))
        new_dict._id = id_
        return new_dict

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return Dict_PB


class DictWrapper(StorableObject):
    def __init__(self, value: object):
        super().__init__(
            data=value,
            id=getattr(value, "id", UID()),
            tags=getattr(value, "tags", []),
            description=getattr(value, "description", ""),
        )
        self.value = value

    def _data_object2proto(self) -> Dict_PB:
        _object2proto = getattr(self.data, "_object2proto", None)
        if _object2proto:
            return _object2proto()

    @staticmethod
    def _data_proto2object(proto: Dict_PB) -> "DictWrapper":
        return Dict._proto2object(proto=proto)

    @staticmethod
    def get_data_protobuf_schema() -> GeneratedProtocolMessageType:
        return Dict_PB

    @staticmethod
    def get_wrapped_type() -> type:
        return Dict

    @staticmethod
    def construct_new_object(
        id: UID,
        data: StorableObject,
        description: Optional[str],
        tags: Optional[List[str]],
    ) -> StorableObject:
        setattr(data, "_id", id)
        data.tags = tags
        data.description = description
        return data


aggressive_set_attr(obj=Dict, name="serializable_wrapper_type", attr=DictWrapper)
