from collections import UserList
from typing import Any
from google.protobuf.reflection import GeneratedProtocolMessageType

from ...decorators import syft_decorator
from .primitive_interface import PyPrimitive
from ...core.common import UID
from ... import serialize, deserialize

from typing import Optional


class List(UserList, PyPrimitive):
    @syft_decorator(typechecking=True, prohibit_args=False)
    def __init__(self, value: Any = None, uid: Optional[UID] = None):
        if value is None:
            value = []

        UserList.__init__(self, value)

        self._id: UID = UID() if uid is None else uid

    @property
    def id(self) -> UID:
        """We reveal PyPrimitive.id as a property to discourage users and
        developers of Syft from modifying .id attributes after an object
        has been initialized.

        :return: returns the unique id of the object
        :rtype: UID
        """
        return self._id

    # @syft_decorator(typechecking=True, prohibit_args=False)
    # def __eq__(self, other: Any) -> PyPrimitive:
    #     res = super(Int, self).__eq__(other)
    #     return PrimitiveFactory.generate_primitive(value=res)

    # @syft_decorator(typechecking=True)
    # def _object2proto(self) -> Int_PB:
    #     int_pb = Int_PB()
    #     int_pb.data = self
    #     int_pb.id.CopyFrom(serialize(self.id))
    #     return int_pb

    # @staticmethod
    # def _proto2object(proto: Int_PB) -> "Int":
    #     int_id: UID = deserialize(blob=proto.id)
    #     return Int(value=proto.data, id=int_id)

    # @staticmethod
    # def get_protobuf_schema() -> GeneratedProtocolMessageType:
    #     return Int_PB
