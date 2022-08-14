# stdlib
from enum import Enum
from typing import Any
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


class StaticAttributeAction(Enum):
    SET = 1
    GET = 2


@serializable(recursive_serde=True)
class GetSetStaticAttributeAction(ImmediateActionWithoutReply):
    __attr_allowlist__ = [
        "path",
        "id_at_location",
        "address",
        "id",
        "action",
        "set_arg",
    ]

    __serde_overrides__ = {
        "action": (
            lambda status: int(status.value),
            lambda int_status: StaticAttributeAction(int(int_status)),
        )
    }

    def __init__(
        self,
        path: str,
        id_at_location: UID,
        address: Address,
        action: StaticAttributeAction,
        msg_id: Optional[UID] = None,
        set_arg: Optional[Any] = None,
    ):
        super().__init__(address, msg_id=msg_id)
        self.path = path
        self.id_at_location = id_at_location
        self.action = action
        self.set_arg = set_arg

    def intersect_keys(
        self,
        left: Dict[VerifyKey, Optional[UID]],
        right: Dict[VerifyKey, Optional[UID]],
    ) -> Dict[VerifyKey, Optional[UID]]:
        return RunClassMethodAction.intersect_keys(left, right)

    def execute_action(self, node: AbstractNode, verify_key: VerifyKey) -> None:
        static_attribute_solver = node.lib_ast.query(self.path)

        if self.action == StaticAttributeAction.SET:
            if self.set_arg is None:
                raise ValueError("MAKE PROPER SCHEMA")

            resolved_arg = node.store.get(key=self.set_arg.id_at_location)
            result = static_attribute_solver.solve_set_value(resolved_arg)
        elif self.action == StaticAttributeAction.GET:
            result = static_attribute_solver.solve_get_value()
        else:
            raise ValueError(f"{self.action} not a valid action!")

        if lib.python.primitive_factory.isprimitive(value=result):
            result = lib.python.primitive_factory.PrimitiveFactory.generate_primitive(
                value=result, id=self.id_at_location
            )
        else:
            if hasattr(result, "id"):
                try:
                    if hasattr(result, "_id"):
                        # set the underlying id
                        result._id = self.id_at_location
                    else:
                        result.id = self.id_at_location

                    if result.id != self.id_at_location:
                        raise AttributeError("IDs don't match")
                except AttributeError as e:
                    err = f"Unable to set id on result {type(result)}. {e}"
                    raise Exception(err)

        if not isinstance(result, StorableObject):
            result = StorableObject(
                id=self.id_at_location,
                data=result,
            )

        node.store[self.id_at_location] = result
