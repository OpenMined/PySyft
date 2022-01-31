# stdlib
from typing import Any
from typing import Dict
from typing import Optional

# third party
from nacl.encoding import HexEncoder
from nacl.signing import VerifyKey
from typing_extensions import final

# relative
from .....common.serde.serializable import serializable
from ....abstract.node_service_interface import NodeServiceInterface
from ...exceptions import AuthorizationError
from ...node_table.utils import model_to_json
from ..generic_payload.messages import GenericPayloadMessage
from ..generic_payload.messages import GenericPayloadMessageWithReply
from ..generic_payload.messages import GenericPayloadReplyMessage


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
