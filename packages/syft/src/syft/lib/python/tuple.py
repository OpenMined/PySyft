# stdlib
from typing import Any
from typing import Optional
from typing import Union

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from typing_extensions import SupportsIndex

# syft absolute
import syft as sy

# relative
from ...core.common import UID
from ...core.common.serde.serializable import serializable
from ...proto.lib.python.tuple_pb2 import Tuple as Tuple_PB
from .iterator import Iterator
from .primitive_factory import PrimitiveFactory
from .primitive_factory import isprimitive
from .primitive_interface import PyPrimitive
from .slice import Slice
from .types import SyPrimitiveRet
from .util import downcast
from .util import upcast


class TupleIterator(Iterator):
    pass


@serializable()
class Tuple(tuple, PyPrimitive):
    def __init__(self, *args: Any):
        self._id = UID()

    @property
    def id(self) -> UID:
        """We reveal PyPrimitive.id as a property to discourage users and
        developers of Syft from modifying .id attributes after an object
        has been initialized.

        :return: returns the unique id of the object
        :rtype: UID
        """
        return self._id

    def upcast(self) -> tuple:
        # recursively upcast
        return tuple(upcast(v) for v in self)

    def __new__(cls, *args: Any) -> "Tuple":
        return super(Tuple, cls).__new__(Tuple, *args)

    def __add__(self, other: Any) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(value=super().__add__(other))

    def __contains__(self, item: Any) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(value=super().__contains__(item))

    def __eq__(self, other: Any) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(value=super().__eq__(other))

    def __hash__(self) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(value=super().__hash__())

    def __ne__(self, other: Any) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(value=super().__ne__(other))

    def __ge__(self, other: Any) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(value=super().__ge__(other))

    def __gt__(self, other: Any) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(value=super().__gt__(other))

    def __le__(self, other: Any) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(value=super().__le__(other))

    def __lt__(self, other: Any) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(value=super().__lt__(other))

    def __mul__(self, other: Any) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(value=super().__mul__(other))

    def __rmul__(self, other: Any) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(value=super().__rmul__(other))

    def __len__(self) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(value=super().__len__())

    def __getitem__(self, key: Union[int, slice, Slice, SupportsIndex]) -> Any:
        if isinstance(key, Slice):
            key = key.upcast()
        value = super().__getitem__(key)
        if isprimitive(value=value):
            return PrimitiveFactory.generate_primitive(value=value)
        else:
            # we can have torch.Tensor and other types
            return value

    def count(self, __value: Any) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(value=super().count(__value))

    def index(
        self, __value: Any, __start: Any = ..., __stop: Any = ...
    ) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(value=super().index(__value, __stop))

    def __iter__(self, max_len: Optional[int] = None) -> TupleIterator:
        return TupleIterator(self, max_len=max_len)

    def _object2proto(self) -> Tuple_PB:
        id_ = sy.serialize(obj=self.id)
        downcasted = [downcast(value=element) for element in self]
        data = [sy.serialize(obj=element, to_bytes=True) for element in downcasted]
        return Tuple_PB(id=id_, data=data)

    @staticmethod
    def _proto2object(proto: Tuple_PB) -> "Tuple":
        id_: UID = sy.deserialize(blob=proto.id)
        value = [
            upcast(sy.deserialize(blob=element, from_bytes=True))
            for element in proto.data
        ]
        new_list = Tuple(value)
        new_list._id = id_
        return new_list

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return Tuple_PB
