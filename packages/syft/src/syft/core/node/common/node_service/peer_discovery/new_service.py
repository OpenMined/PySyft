# stdlib
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Type

# third party
from nacl.encoding import HexEncoder
from nacl.signing import VerifyKey
from typing_extensions import final

# relative
from .....common.serde.serializable import serializable
from ....abstract.node_service_interface import NodeServiceInterface
from ...exceptions import AuthorizationError
from ...node_table.utils import model_to_json
from ..auth import service_auth
from ..generic_payload.messages import GenericPayloadMessage
from ..generic_payload.messages import GenericPayloadMessageWithReply
from ..generic_payload.messages import GenericPayloadReplyMessage
from ..node_service import NodeService


@serializable(recursive_serde=True)
@final
class GetMessage(GenericPayloadMessage):
    ...


@serializable(recursive_serde=True)
@final
class GetReplyMessage(GenericPayloadReplyMessage):
    ...


@serializable(recursive_serde=True)
@final
class GetUserMessage(GenericPayloadMessageWithReply):
    message_type = GetMessage
    message_reply_type = GetReplyMessage

    def run(
        self, node: NodeServiceInterface, verify_key: Optional[VerifyKey] = None
    ) -> Dict[str, Any]:

        # TODO: Segregate permissions to a different level (make it composable)
        _allowed = node.users.can_triage_requests(verify_key=verify_key)
        if not _allowed:
            raise AuthorizationError(
                "get_user_msg You're not allowed to get User information!"
            )
        else:
            # Extract User Columns
            user = node.users.first(id=self.kwargs["user_id"])
            _msg = model_to_json(user)

            # Use role name instead of role ID.
            _msg["role"] = node.roles.first(id=_msg["role"]).name

            # Remove private key
            del _msg["private_key"]

            # Get budget spent
            _msg["budget_spent"] = node.acc.user_budget(
                user_key=VerifyKey(user.verify_key.encode("utf-8"), encoder=HexEncoder)
            )
            return _msg


class DomainServiceRegistry:
    _domain_registry = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._domain_registry.append(cls)

    @classmethod
    def get_registered_services(cls):
        """dict: Returns map of service inheriting this class."""
        return list(cls._domain_registry)


class NetworkServiceRegistry:
    """A class for registering the services attached to the domain client."""

    _network_registry = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._network_registry.append(cls)

    @classmethod
    def get_registered_services(cls):
        """dict: Returns map of service inheriting this class."""
        return list(cls._network_registry)


class UserService(NetworkServiceRegistry, DomainServiceRegistry, NodeService):
    @staticmethod
    @service_auth(guests_welcome=True)
    def process(
        node: NodeServiceInterface,
        msg: GetMessage,
        verify_key: Optional[VerifyKey] = None,
    ) -> GetUserMessage:

        result = msg.payload.run(node=node, verify_key=verify_key)
        return GetUserMessage(kwargs=result).back_to(address=msg.reply_to)

    @staticmethod
    def message_handler_types() -> List[Type[GetMessage]]:
        return [GetMessage]
