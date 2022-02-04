# stdlib
from typing import Any
from typing import Dict
from typing import Optional

# third party
from nacl.encoding import HexEncoder
from nacl.signing import VerifyKey
from typing_extensions import final

# relative
from .....common.message import ImmediateSyftMessage
from .....common.serde.serializable import serializable
from ....abstract.node_service_interface import NodeServiceInterface
from ....domain.registry import DomainMessageRegistry
from ...exceptions import AuthorizationError
from ...node_table.utils import model_to_json
from ..generic_payload.messages import GenericPayloadMessage
from ..generic_payload.messages import GenericPayloadMessageWithReply
from ..generic_payload.messages import GenericPayloadReplyMessage
from ..generic_payload.syft_message import SyftMessage


@serializable(recursive_serde=True)
@final
class GetUserMessage(SyftMessage, DomainMessageRegistry):
    def run(
        self, node: NodeServiceInterface, verify_key: Optional[VerifyKey] = None
    ) -> ImmediateSyftMessage:
        if not verify_key:
            return {}

        # TODO: Segregate permissions to a different level (make it composable)
        _allowed = node.users.can_triage_requests(verify_key=verify_key)  # type: ignore
        if not _allowed:
            raise AuthorizationError(
                "get_user_msg You're not allowed to get User information!"
            )
        else:
            # Extract User Columns
            user = node.users.first(id=self.kwargs["user_id"])  # type: ignore
            _msg = model_to_json(user)

            # Use role name instead of role ID.
            _msg["role"] = node.roles.first(id=_msg["role"]).name  # type: ignore

            # Remove private key
            del _msg["private_key"]

            # Get budget spent
            _msg["budget_spent"] = node.acc.user_budget(  # type: ignore
                user_key=VerifyKey(user.verify_key.encode("utf-8"), encoder=HexEncoder)
            )
            return _msg


@serializable(recursive_serde=True)
@final
class GetUsersMessage(SyftMessage, DomainMessageRegistry):
    def run(
        self, node: NodeServiceInterface, verify_key: Optional[VerifyKey] = None
    ) -> ImmediateSyftMessage:
        # Check key permissions
        _allowed = node.users.can_triage_requests(verify_key=verify_key)
        if not _allowed:
            raise AuthorizationError(
                "get_all_users_msg You're not allowed to get User information!"
            )

        # Get All Users
        users = node.users.all()
        _msg = []
        for user in users:
            _user_json = model_to_json(user)
            # Use role name instead of role ID.
            _user_json["role"] = node.roles.first(id=_user_json["role"]).name

            # Remove private key
            del _user_json["private_key"]

            # Remaining Budget
            # TODO:
            # Rename it from budget_spent to remaining budget
            _user_json["budget_spent"] = node.acc.get_remaining_budget(  # type: ignore
                user_key=VerifyKey(user.verify_key.encode("utf-8"), encoder=HexEncoder),
                returned_epsilon_is_private=False,
            )
            _msg.append(_user_json)

            return _msg

