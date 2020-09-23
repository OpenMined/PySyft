# stdlib
from collections import UserDict
from collections.abc import Iterable
from collections.abc import Mapping
from typing import Any
from typing import List
from typing import Optional
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


class Dict(UserDict, PyPrimitive):
    @syft_decorator(typechecking=False, prohibit_args=False)
    # the incoming types to UserDict __init__ are overloaded and weird
    # see https://github.com/python/cpython/blob/master/Lib/collections/__init__.py
    def __init__(self, dict: Any = None, /, **kwargs: Any) -> None:
        self.data = {}

        # there is so much bad juju going on here but unfortunately, its all to work
        # around the deprecated ability to pass {"dict": {}} as params to a dict as well
        # as support a multitude of other types of Iterables. Annoyingly subclassing
        # from UserDict and using its __init__ doesnt fix this automatically.
        if dict is not None:
            # Dont be fooled by the variable name this might not be a dict and probably
            # means we have an iterable of positional args coming in
            self.update(dict)
        if kwargs:
            # Here we have a dict, but we need to handle several different cases
            # 1) Where kwargs contains a key called "dict" which is itself an Iterable.
            # There is different behavior between dict and UserDict:
            # >>> l = [('one', 1), ('two', 2)]
            #
            # >>> dict(dict=l)
            # {'dict': [('one', 1), ('two', 2)]}
            #
            # >>> UserDict(dict=l)
            # {'one': 1, 'two': 2}
            #
            # 2) All sorts of other types can be sent through, lists, scalars and they
            # need to be handled in different ways.
            #
            # This is passing both the dict and UserDict tests with only modifications
            # to the lack of support for reversed. We even raise the DeprecationWarning.
            if "dict" in kwargs.keys():
                if issubclass(type(kwargs["dict"]), Iterable):
                    iter_kwargs = kwargs["dict"]
                    # Its pretty clear why dict={} is a mistake as it causes
                    # all of this havoc, just consider for a moment that the
                    # type builtins.dict is no longer usable within this function
                    warnings.warn(
                        "Passing 'dict' as keyword argument is deprecated",
                        DeprecationWarning,
                        2,
                    )
                    if issubclass(type(iter_kwargs), Mapping):
                        self.update(**iter_kwargs)
                        for k, v in kwargs.items():
                            if k != "dict":
                                self.update({k: v})
                    else:
                        self.update(iter_kwargs)
                else:
                    # the result is not an interable to it will work
                    self.update(**kwargs)

            else:
                self.update(kwargs)

        # finally lets add our UID
        self._id: UID = kwargs["id"] if "id" in kwargs else UID()

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

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> Dict_PB:
        id_ = serialize(obj=self.id)
        keys = list(self.data.keys())
        values = [serialize(obj=element) for element in self.data.values()]
        return Dict_PB(id=id_, keys=keys, values=values)

    @staticmethod
    @syft_decorator(typechecking=True)
    def _proto2object(proto: Dict_PB) -> "Dict":
        id_: UID = deserialize(blob=proto.id)
        values = [deserialize(blob=element) for element in proto.values]
        new_dict = Dict(dict(zip(proto.keys, values)))
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
