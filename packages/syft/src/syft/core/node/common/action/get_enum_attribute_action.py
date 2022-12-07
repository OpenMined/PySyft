# stdlib
from typing import Dict
from typing import Optional

# third party
from nacl.signing import VerifyKey

# relative
from ..... import lib
from ....common.serde.serializable import serializable
from ....common.uid import UID
from ....io.address import Address
from ....store.storeable_object import StorableObject
from ...abstract.node import AbstractNode
from .common import ImmediateActionWithoutReply
from .run_class_method_action import RunClassMethodAction


@serializable(recursive_serde=True)
class EnumAttributeAction(ImmediateActionWithoutReply):
    __attr_allowlist__ = ["path", "id_at_location", "address", "id"]

    def __init__(
        self,
        path: str,
        id_at_location: UID,
        address: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address, msg_id=msg_id)
        self.id_at_location = id_at_location
        self.path = path

    def intersect_keys(
        self,
        left: Dict[VerifyKey, Optional[UID]],
        right: Dict[VerifyKey, Optional[UID]],
    ) -> Dict[VerifyKey, Optional[UID]]:
        return RunClassMethodAction.intersect_keys(left, right)

    def execute_action(self, node: AbstractNode, verify_key: VerifyKey) -> None:
        enum_attribute = node.lib_ast.query(self.path)
        result = enum_attribute.solve_get_enum_attribute().value
        result = lib.python.primitive_factory.PrimitiveFactory.generate_primitive(
            value=result, id=self.id_at_location
        )

        result = StorableObject(
            id=self.id_at_location,
            data=result,
        )

        node.store[self.id_at_location] = result
