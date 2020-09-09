# stdlib
from collections import UserList
from typing import Any
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType

# syft relative
from ... import deserialize
from ... import serialize
from ...core.common import UID
from ...decorators import syft_decorator
from ...proto.lib.python.list_pb2 import List as List_PB
from .primitive_interface import PyPrimitive


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

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> List_PB:
        id_ = serialize(obj=self.id)
        data = [serialize(obj=element) for element in self.data]
        return List_PB(id=id_, data=data)

    @staticmethod
    @syft_decorator(typechecking=True)
    def _proto2object(proto: List_PB) -> "List":
        id_: UID = deserialize(blob=proto.id)
        value = [deserialize(blob=element) for element in proto.data]
        return List(value=value, uid=id_)

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return List_PB
