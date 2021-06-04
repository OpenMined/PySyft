# stdlib
from datetime import datetime
import secrets
from typing import List
from typing import Type
from typing import Union

# third party
from flask import current_app as app
from nacl.encoding import HexEncoder
from nacl.signing import SigningKey
from nacl.signing import VerifyKey
from syft.core.common.message import ImmediateSyftMessageWithReply

# syft relative
from syft.core.node.abstract.node import AbstractNode
from syft.core.node.common.service.auth import service_auth
from syft.core.node.common.service.node_service import ImmediateNodeServiceWithReply
from syft.core.node.common.service.node_service import ImmediateNodeServiceWithoutReply
from syft.grid.messages.setup_messages import CreateInitialSetUpMessage
from syft.grid.messages.setup_messages import CreateInitialSetUpResponse
from syft.grid.messages.setup_messages import GetSetUpMessage
from syft.grid.messages.setup_messages import GetSetUpResponse

# grid relative
from ...core.database.environment.environment import states
from ...core.infrastructure import AWS_Serverfull
from ...core.infrastructure import AWS_Serverless
from ...core.infrastructure import Config
from ...core.infrastructure import Provider
from ..database.setup.setup import SetupConfig
from ..database.utils import model_to_json
from ..exceptions import AuthorizationError
from ..exceptions import InvalidParameterValueError
from ..exceptions import MissingRequestKeyError
from ..exceptions import OwnerAlreadyExistsError


def create_initial_setup(
    msg: CreateInitialSetUpMessage, node: AbstractNode, verify_key: VerifyKey
) -> CreateInitialSetUpResponse:
    # Should not run if Domain has an owner
    if len(node.users):
        raise OwnerAlreadyExistsError

    _email = msg.content.get("email", None)
    _password = msg.content.get("password", None)

    # Get Payload Content
    configs = {
        "domain_name": msg.content.get("domain_name", ""),
        "private_key": msg.content.get("private_key", ""),
        "aws_credentials": msg.content.get("aws_credentials", ""),
        "gcp_credentials": msg.content.get("gcp_credentials", ""),
        "azure_credentials": msg.content.get("azure_credentials", ""),
        "cache_strategy": msg.content.get("cache_strategy", ""),
        "replicate_db": msg.content.get("replicate_db", False),
        "auto_scale": msg.content.get("auto_scale", False),
        "tensor_expiration_policy": msg.content.get("tensor_expiration_policy", 0),
        "allow_user_signup": msg.content.get("allow_user_signup", False),
    }

    _current_user_id = msg.content.get("current_user", None)

    users = node.users

    if not _current_user_id:
        try:
            _current_user_id = users.first(
                verify_key=verify_key.encode(encoder=HexEncoder).decode("utf-8")
            ).id
        except Exception:
            pass

    _admin_role = node.roles.first(name="Owner")

    _mandatory_request_fields = _email and _password and configs["domain_name"]

    # Check if email/password/node_name fields are empty
    if not _mandatory_request_fields:
        raise MissingRequestKeyError(
            message="Invalid request payload, empty fields (email/password/domain_name)!"
        )

    config_obj = SetupConfig(**configs)

    # Change Node Name
    node.name = config_obj.domain_name

    # Change Node Root Key (if requested)
    if config_obj.private_key != "":
        try:
            private_key = SigningKey(config_obj.encode("utf-8"), encoder=HexEncoder)
        except Exception:
            raise InvalidParameterValueError("Invalid Signing Key!")
        node.root_key = private_key
        node.verify_key = private_key.verify_key

    # Create Admin User
    _node_private_key = node.signing_key.encode(encoder=HexEncoder).decode("utf-8")
    _verify_key = node.signing_key.verify_key.encode(encoder=HexEncoder).decode("utf-8")
    _admin_role = node.roles.first(name="Owner")
    _user = users.signup(
        email=_email,
        password=_password,
        role=_admin_role.id,
        private_key=_node_private_key,
        verify_key=_verify_key,
    )

    # Final status / message
    final_msg = "Running initial setup!"
    node.setup.register(**configs)
    return CreateInitialSetUpResponse(
        address=msg.reply_to,
        status_code=200,
        content={"message": final_msg},
    )


def get_setup(
    msg: GetSetUpMessage, node: AbstractNode, verify_key: VerifyKey
) -> GetSetUpResponse:

    _current_user_id = msg.content.get("current_user", None)

    users = node.users

    if not _current_user_id:
        try:
            _current_user_id = users.first(
                verify_key=verify_key.encode(encoder=HexEncoder).decode("utf-8")
            ).id
        except Exception:
            pass

    if users.role(user_id=_current_user_id).name != "Owner":
        raise AuthorizationError("You're not allowed to get setup configs!")
    else:
        _setup = model_to_json(node.setup.first(domain_name=node.name))

    return GetSetUpResponse(
        address=msg.reply_to,
        status_code=200,
        content=_setup,
    )


class SetUpService(ImmediateNodeServiceWithReply):

    msg_handler_map = {
        CreateInitialSetUpMessage: create_initial_setup,
        GetSetUpMessage: get_setup,
    }

    @staticmethod
    @service_auth(guests_welcome=True)
    def process(
        node: AbstractNode,
        msg: Union[
            CreateInitialSetUpMessage,
            GetSetUpMessage,
        ],
        verify_key: VerifyKey,
    ) -> Union[CreateInitialSetUpResponse, GetSetUpResponse,]:
        return SetUpService.msg_handler_map[type(msg)](
            msg=msg, node=node, verify_key=verify_key
        )

    @staticmethod
    def message_handler_types() -> List[Type[ImmediateSyftMessageWithReply]]:
        return [
            CreateInitialSetUpMessage,
            GetSetUpMessage,
        ]
