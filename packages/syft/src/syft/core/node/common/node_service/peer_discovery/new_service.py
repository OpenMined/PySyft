"""
Service
    - create
    - del
    - update/patch
    - get

    mapping: Message: function_type


WithReply: Blocking and return
WithoutReply: Return None

Unify with and without reply ->  Services are `reply` agnostic

Composable Permissions
"""

"""
from venv import create
from webbrowser import get

from numpy import delete


UserService:
- create
- delete
- update
- get

GetService::::::::::
- users, datasets, roles


GetMessage:
  - def get

UpdateMessage:
  - def u

DeleteMessage:
  - process


UserMessage(GetMessage, UpdateMessage, DeleteMessage):
 - get_message  = GetMessage
 - update_message = UpdateMessage

 UserMessage()
  user_messages = [GetMessage, UpdateMessage]

"""
# future
from __future__ import annotations

# stdlib
from abc import ABCMeta
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import cast

# third party
from nacl.encoding import HexEncoder
from nacl.signing import VerifyKey
from typing_extensions import final

# relative
from .....common.serde.serializable import serializable
from ....abstract.node_service_interface import NodeServiceInterface
from ...exceptions import AuthorizationError
from ...node_table.node import Node as NodeRow
from ...node_table.node_route import NodeRoute as NodeRouteRow
from ...node_table.utils import model_to_json
from ..generic_payload.messages import GenericPayloadMessage
from ..generic_payload.messages import GenericPayloadMessageWithReply
from ..generic_payload.messages import GenericPayloadReplyMessage


@serializable(recursive_serde=True)
@final
class GetMessage(GenericPayloadMessage):
    message_protocol = "GET"
    ...


@serializable(recursive_serde=True)
@final
class GetReplyMessage(GenericPayloadReplyMessage):
    ...


@serializable(recursive_serde=True)
@final
class GetUserMessageWithReply(GenericPayloadMessageWithReply):
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


class DomainServiceRegistry(ABCMeta):
    """A class for registering the services attached to the domain client."""

    _service_registry = {}

    def __new__(cls, *args, **kwargs):
        new_cls = type.__new__(cls, *args, **kwargs)
        if hasattr(new_cls, "__name__"):
            service_name = getattr(new_cls, "__name__")
            cls._service_registry[service_name] = new_cls
        return super(DomainServiceRegistry, cls).__new__(cls, *args, **kwargs)

    @classmethod
    def get_registered_services(cls):
        """dict: Returns map of service inheriting this class."""
        return dict(cls._service_registry)


class NetworkServiceRegistry(ABCMeta):
    """A class for registering the services attached to the domain client."""

    _service_registry = {}

    def __new__(cls, *args, **kwargs):
        new_cls = type.__new__(cls, *args, **kwargs)
        if hasattr(new_cls, "__name__"):
            service_name = getattr(new_cls, "__name__")
            cls._service_registry[service_name] = new_cls
        return super(NetworkServiceRegistry, cls).__new__(cls, *args, **kwargs)

    @classmethod
    def get_registered_services(cls):
        """dict: Returns map of service inheriting this class."""
        return dict(cls._service_registry)


class UserService(
    DomainServiceRegistry, NetworkServiceRegistry, GenericPayloadMessageWithReply
):
    pass
