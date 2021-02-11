# stdlib
from typing import List
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType

# syft absolute
import syft
from syft.core.node.common.action.common import Action
from syft.proto.core.node.common.action.action_pb2 import Action as Action_PB

# syft relative
from ...core.common.uid import UID
from ...core.store.storeable_object import StorableObject
from ...logger import warning
from ...util import aggressive_set_attr


class ActionWrapper(StorableObject):
    def __init__(self, value: object):
        super().__init__(
            data=value,
            id=getattr(value, "id", UID()),
            tags=getattr(value, "tags", []),
            description=getattr(value, "description", ""),
        )
        self.value = value

    def _data_object2proto(self) -> Action_PB:
        proto = self.value._object2proto()
        return proto

    @staticmethod
    def _data_proto2object(proto: Action_PB) -> Action:
        # obj = syft.deserialize(blob=proto)
        return Action._proto2object(proto)

    @staticmethod
    def get_data_protobuf_schema() -> GeneratedProtocolMessageType:
        return Action_PB

    @staticmethod
    def get_wrapped_type() -> type:
        return Action

    # @staticmethod
    # def construct_new_object(
    #     id: UID,
    #     data: StorableObject,
    #     description: Optional[str],
    #     tags: Optional[List[str]],
    # ) -> StorableObject:
    #     data.id = id
    #     data.tags = tags
    #     data.description = description
    #     return data


aggressive_set_attr(obj=Action, name="serializable_wrapper_type", attr=ActionWrapper)
