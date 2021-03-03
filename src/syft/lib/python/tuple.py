# stdlib
from typing import Any
from typing import Optional
from typing import Union

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType

# syft relative
from ... import deserialize
from ... import serialize
from ...core.common import UID
from ...core.common.serde.serializable import bind_protobuf
from ...proto.lib.python.tuple_pb2 import Tuple as Tuple_PB
from .iterator import TemplateableIterator
from .primitive_factory import PrimitiveFactory
from .primitive_factory import isprimitive
from .primitive_interface import PyPrimitive
from .types import SyPrimitiveRet
from .util import downcast
from .util import upcast


@bind_protobuf
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

    def __getitem__(self, item: Any) -> Union[SyPrimitiveRet, Any]:
        value = super().__getitem__(item)
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

    def __iter__(self, max_len: Optional[int] = None) -> TemplateableIterator:
        return TemplateableIterator(self, max_len=max_len)

    @staticmethod
    def _proto2object(proto: Tuple_PB) -> "Tuple":
        id_: UID = deserialize(blob=proto.id)
        value = [
            upcast((deserialize(blob=element, from_bytes=True)))
            for element in proto.data
        ]
        new_list = Tuple(value)
        new_list._id = id_
        return new_list

    def _object2proto(self) -> Tuple_PB:
        id_ = serialize(obj=self.id)
        downcasted = [downcast(value=element) for element in self]
        data = [serialize(obj=element, to_bytes=True) for element in downcasted]
        return Tuple_PB(id=id_, data=data)

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return Tuple_PB
