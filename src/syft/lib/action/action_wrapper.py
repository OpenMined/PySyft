# stdlib
from typing import List
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType

# syft absolute
import syft
# from syft.core.node.common.action.common import Action
# from syft.proto.core.node.common.action.action_pb2 import Action as Action_PB
from syft.proto.core.node.common.plan.plan_pb2 import Plan as Plan_PB
from syft.core.node.common.plan.plan import Plan

# syft relative
from ...core.common.uid import UID
from ...core.store.storeable_object import StorableObject
from ...logger import warning
from ...util import aggressive_set_attr


class PlanWrapper(StorableObject):
    def __init__(self, value: object):
        super().__init__(
            data=value,
            id=getattr(value, "id", UID()),
            tags=getattr(value, "tags", []),
            description=getattr(value, "description", ""),
        )
        self.value = value

    def _data_object2proto(self) -> Plan_PB:
        return self.value._object2proto()

    @staticmethod
    def _data_proto2object(proto: Plan_PB) -> Plan:
        return Plan._proto2object(proto)

    @staticmethod
    def get_data_protobuf_schema() -> GeneratedProtocolMessageType:
        return Plan_PB

    @staticmethod
    def get_wrapped_type() -> type:
        return Plan


aggressive_set_attr(obj=Plan, name="serializable_wrapper_type", attr=PlanWrapper)
