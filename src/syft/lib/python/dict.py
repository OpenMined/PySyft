# stdlib
from collections import UserDict
from collections.abc import ItemsView
from collections.abc import KeysView
from collections.abc import ValuesView
from typing import Any
from typing import Iterable
from typing import List
from typing import Optional
from typing import Union
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
from .iterator import Iterator
from .primitive_factory import PrimitiveFactory
from .primitive_factory import isprimitive
from .primitive_interface import PyPrimitive
from .util import SyPrimitiveRet
from .util import downcast
from .util import upcast


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
    def __contains__(self, other: Any) -> SyPrimitiveRet:
        res = super().__contains__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __eq__(self, other: Any) -> SyPrimitiveRet:
        res = super().__eq__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __format__(self, format_spec: str) -> str:
        # python complains if the return value is not str
        res = super().__format__(format_spec)
        return str(res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __ge__(self, other: Any) -> SyPrimitiveRet:
        res = super().__ge__(other)  # type: ignore
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __getitem__(self, key: Any) -> Union[SyPrimitiveRet, Any]:
        res = super().__getitem__(key)
        if isprimitive(value=res):
            return PrimitiveFactory.generate_primitive(value=res)
        else:
            # we can have torch.Tensor and other types
            return res

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __gt__(self, other: Any) -> SyPrimitiveRet:
        res = super().__gt__(other)  # type: ignore
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __hash__(self) -> SyPrimitiveRet:
        res = super().__hash__()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __iter__(self, max_len: Optional[int] = None) -> Iterator:
        return Iterator(super().__iter__(), max_len=max_len)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __le__(self, other: Any) -> SyPrimitiveRet:
        res = super().__le__(other)  # type: ignore
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __len__(self) -> SyPrimitiveRet:
        res = super().__len__()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __lt__(self, other: Any) -> SyPrimitiveRet:
        res = super().__lt__(other)  # type: ignore
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __ne__(self, other: Any) -> SyPrimitiveRet:
        res = super().__ne__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __sizeof__(self) -> SyPrimitiveRet:
        res = super().__sizeof__()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def copy(self) -> SyPrimitiveRet:
        res = super().copy()
        return PrimitiveFactory.generate_primitive(value=res)

    @classmethod
    @syft_decorator(typechecking=True, prohibit_args=False)
    def fromkeys(
        cls, iterable: Iterable, value: Optional[Any] = None
    ) -> SyPrimitiveRet:
        res = super().fromkeys(iterable, value)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def get(self, key: Any, default: Any = None) -> SyPrimitiveRet:
        res = super().get(key, default)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def items(self, max_len: Optional[int] = None) -> Iterator:
        return Iterator(ItemsView(self), max_len=max_len)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def keys(self, max_len: Optional[int] = None) -> Iterator:
        return Iterator(KeysView(self), max_len=max_len)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def values(self, *args: Any, max_len: Optional[int] = None) -> Iterator:
        # this is what the super type does and there is a test in dict_test.py
        # test_values which checks for this so we could disable the test or
        # keep this workaround
        if len(args) > 0:
            raise TypeError("values() takes 1 positional argument but 2 were given")
        return Iterator(ValuesView(self), max_len=max_len)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def pop(self, key: Any, *args: Any) -> SyPrimitiveRet:
        res = super().pop(key, *args)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def popitem(self) -> SyPrimitiveRet:
        res = super().popitem()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def setdefault(self, key: Any, default: Any = None) -> SyPrimitiveRet:
        res = PrimitiveFactory.generate_primitive(value=default)
        res = super().setdefault(key, res)
        return res

    @syft_decorator(typechecking=True, prohibit_args=False)
    def clear(self) -> SyPrimitiveRet:
        # we get the None return and create a SyNone
        # this is to make sure someone doesn't rewrite the method to return nothing
        res = super().clear()  # pylint: disable=E1111
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> Dict_PB:
        id_ = serialize(obj=self.id)
        # serialize to bytes so that we can avoid using StorableObject
        # otherwise we get recursion where the permissions of StorableObject themselves
        # utilise Dict
        keys = [
            serialize(obj=downcast(value=element), to_bytes=True)
            for element in self.data.keys()
        ]
        # serialize to bytes so that we can avoid using StorableObject
        # otherwise we get recursion where the permissions of StorableObject themselves
        # utilise Dict
        values = [
            serialize(obj=downcast(value=element), to_bytes=True)
            for element in self.data.values()
        ]
        return Dict_PB(id=id_, keys=keys, values=values)

    @staticmethod
    @syft_decorator(typechecking=True)
    def _proto2object(proto: Dict_PB) -> "Dict":
        id_: UID = deserialize(blob=proto.id)
        # deserialize from bytes so that we can avoid using StorableObject
        # otherwise we get recursion where the permissions of StorableObject themselves
        # utilise Dict
        values = [
            deserialize(blob=upcast(value=element), from_bytes=True)
            for element in proto.values
        ]
        # deserialize from bytes so that we can avoid using StorableObject
        # otherwise we get recursion where the permissions of StorableObject themselves
        # utilise Dict
        keys = [
            deserialize(blob=upcast(value=element), from_bytes=True)
            for element in proto.keys
        ]
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
