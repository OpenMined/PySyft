# stdlib
from typing import Callable
from typing import Dict
from typing import List
from typing import Type
from typing import Union

# third party
from nacl.encoding import HexEncoder
from nacl.signing import VerifyKey

# syft absolute
from syft.core.common.message import ImmediateSyftMessageWithReply
from syft.core.node.abstract.node import AbstractNode
from syft.core.node.common.node_service.auth import service_auth
from syft.core.node.common.node_service.node_service import (
    ImmediateNodeServiceWithReply,
)

# relative
from ......logger import traceback_and_raise
from .....common import UID
from .....io.location import SpecificLocation
from ...exceptions import AuthorizationError
from ...exceptions import MissingRequestKeyError
from ...exceptions import OwnerAlreadyExistsError
from ...node_table.utils import model_to_json
from ..success_resp_message import SuccessResponseMessage
from .node_setup_messages import CreateInitialSetUpMessage
from .node_setup_messages import GetSetUpMessage
from .node_setup_messages import GetSetUpResponse


def set_node_uid(node: AbstractNode) -> None:
    try:
        setup = node.setup.first()
    except Exception as e:
        print("Missing Setup Table entry", e)

    try:
        node_id = UID.from_string(setup.node_id)
    except Exception as e:
        print(f"Invalid Node UID in Setup Table. {setup.node_id}")
        raise e

    location = SpecificLocation(name=setup.domain_name, id=node_id)
    # TODO: Fix with proper isinstance when the class will import
    if type(node).__name__ == "Domain":
        node.domain = location
    elif type(node).__name__ == "Network":
        node.network = location
    print(f"Finished setting Node UID. {location}")


def create_initial_setup(
    msg: CreateInitialSetUpMessage, node: AbstractNode, verify_key: VerifyKey
) -> SuccessResponseMessage:

    # 1 - Should not run if Node has an owner
    if len(node.users):
        set_node_uid(node=node)  # make sure the node always has the same UID
        raise OwnerAlreadyExistsError

    # 2 - Check if email/password/node_name fields are empty
    _mandatory_request_fields = msg.email and msg.password and msg.domain_name
    if not _mandatory_request_fields:
        raise MissingRequestKeyError(
            message="Invalid request payload, empty fields (email/password/domain_name)!"
        )

    # 3 - Change Node Name
    node.name = msg.domain_name

    # 4 - Create Admin User
    _node_private_key = node.signing_key.encode(encoder=HexEncoder).decode("utf-8")
    _verify_key = node.signing_key.verify_key.encode(encoder=HexEncoder).decode("utf-8")
    _admin_role = node.roles.owner_role
    _ = node.users.signup(
        name=msg.name,
        email=msg.email,
        password=msg.password,
        role=_admin_role.id,
        budget=msg.budget,
        private_key=_node_private_key,
        verify_key=_verify_key,
    )

    # 5 - Save Node SetUp Configs
    try:
        node_id = node.target_id.id
        node.setup.register(domain_name=msg.domain_name, node_id=node_id.no_dash)
    except Exception as e:
        print("Failed to save setup to database", e)

    return SuccessResponseMessage(
        address=msg.reply_to,
        resp_msg="Running initial setup!",
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
        except Exception as e:
            traceback_and_raise(e)

    if users.role(user_id=_current_user_id).name != "Owner":
        raise AuthorizationError("You're not allowed to get setup configs!")
    else:
        _setup = model_to_json(node.setup.first(domain_name=node.name))

    return GetSetUpResponse(
        address=msg.reply_to,
        status_code=200,
        content=_setup,
    )


class NodeSetupService(ImmediateNodeServiceWithReply):

    msg_handler_map: Dict[type, Callable] = {
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
    ) -> Union[SuccessResponseMessage, GetSetUpResponse]:
        return NodeSetupService.msg_handler_map[type(msg)](
            msg=msg, node=node, verify_key=verify_key
        )

    @staticmethod
    def message_handler_types() -> List[Type[ImmediateSyftMessageWithReply]]:
        return [
            CreateInitialSetUpMessage,
            GetSetUpMessage,
        ]
