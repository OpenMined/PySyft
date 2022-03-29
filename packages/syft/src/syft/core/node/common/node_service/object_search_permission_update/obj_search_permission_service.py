# stdlib
from typing import List
from typing import Type

# third party
from nacl.signing import VerifyKey

# relative
from .....common.group import VERIFYALL
from ....abstract.node import AbstractNode
from ..auth import AuthorizationException
from ..auth import service_auth
from ..node_service import ImmediateNodeServiceWithoutReply
from .obj_search_permission_messages import ObjectSearchPermissionUpdateMessage


class ImmediateObjectSearchPermissionUpdateService(ImmediateNodeServiceWithoutReply):
    @staticmethod
    @service_auth(root_only=True)
    def process(
        node: AbstractNode,
        msg: ObjectSearchPermissionUpdateMessage,
        verify_key: VerifyKey,
    ) -> None:
        storable_object = node.store.get(msg.target_object_id, proxy_only=True)
        if (
            verify_key != node.root_verify_key
            or verify_key not in storable_object.read_permissions
        ):
            log = (
                f"You do not have permission to update Object with ID: {msg.target_object_id}"
                + "Please submit a request."
            )
            raise AuthorizationException(log)
        target_verify_key = msg.target_verify_key or VERIFYALL
        if msg.add_instead_of_remove:
            storable_object.search_permissions[target_verify_key] = msg.id
        else:
            storable_object.search_permissions.pop(target_verify_key, None)

        node.store[msg.target_object_id] = storable_object

    @staticmethod
    def message_handler_types() -> List[Type[ObjectSearchPermissionUpdateMessage]]:
        return [ObjectSearchPermissionUpdateMessage]
