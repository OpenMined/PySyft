# stdlib
from datetime import datetime
from json import dumps
from json import loads
from typing import Callable
from typing import Dict
from typing import List
from typing import Type
from typing import Union

# third party
from nacl.encoding import HexEncoder
from nacl.signing import VerifyKey

# relative
from .....common import UID
from .....common.message import ImmediateSyftMessageWithReply
from .....io.location import SpecificLocation
from ....domain_interface import DomainInterface
from ...exceptions import AuthorizationError
from ...exceptions import MissingRequestKeyError
from ...exceptions import OwnerAlreadyExistsError
from ...node_table.utils import model_to_json
from ..auth import service_auth
from ..node_service import ImmediateNodeServiceWithReply
from ..success_resp_message import SuccessResponseMessage
from .node_setup_messages import CreateInitialSetUpMessage
from .node_setup_messages import GetSetUpMessage
from .node_setup_messages import GetSetUpResponse
from .node_setup_messages import UpdateSetupMessage
from .node_setup_messages import UpdateSetupResponse


def set_node_uid(node: DomainInterface) -> None:
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
    msg: CreateInitialSetUpMessage, node: DomainInterface, verify_key: VerifyKey
) -> SuccessResponseMessage:

    print("Performing initial setup...")

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
    signing_key = msg.signing_key

    # convert to hex
    _node_private_key = signing_key.encode(encoder=HexEncoder).decode("utf-8")  # type: ignore
    _verify_key = signing_key.verify_key.encode(encoder=HexEncoder).decode("utf-8")  # type: ignore

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
        node.setup.register_once(
            domain_name=msg.domain_name,
            node_id=node_id.no_dash,
            deployed_on=datetime.now(),
            signing_key=_node_private_key,
        )
    except Exception as e:
        print("Failed to save setup to database", e)

    return SuccessResponseMessage(
        address=msg.reply_to,
        resp_msg="Running initial setup!",
    )


def get_setup(
    msg: GetSetUpMessage, node: DomainInterface, verify_key: VerifyKey
) -> GetSetUpResponse:

    _setup = model_to_json(node.setup.first(domain_name=node.name))
    _setup["tags"] = loads(_setup["tags"])
    # TODO: Make this a little more defensive so we dont accidentally spill secrets
    # from node.settings. Perhaps we should add a public settings interface
    _setup["use_blob_storage"] = getattr(node.settings, "USE_BLOB_STORAGE", False)
    # Remove it before sending setup's response
    del _setup["signing_key"]

    if node.network:
        _setup["domains"] = len(node.node.all())
    return GetSetUpResponse(
        address=msg.reply_to,
        content=_setup,
    )


def update_settings(
    msg: UpdateSetupMessage, node: DomainInterface, verify_key: VerifyKey
) -> UpdateSetupResponse:
    if node.users.role(verify_key=verify_key).id == node.roles.owner_role.id:
        if msg.domain_name:
            node.name = msg.domain_name

        node.setup.update(
            domain_name=node.name,
            description=msg.description,
            daa=msg.daa,
            contact=msg.contact,
            daa_document=msg.daa_document,
            tags=dumps(msg.tags),
        )
    else:
        raise AuthorizationError("You're not allowed to get setup configs!")

    return UpdateSetupResponse(
        address=msg.reply_to,
        content={"message": "Node settings have been updated successfully!"},
    )


class NodeSetupService(ImmediateNodeServiceWithReply):

    msg_handler_map: Dict[type, Callable] = {
        CreateInitialSetUpMessage: create_initial_setup,
        GetSetUpMessage: get_setup,
        UpdateSetupMessage: update_settings,
    }

    @staticmethod
    @service_auth(guests_welcome=True)
    def process(
        node: DomainInterface,
        msg: Union[
            CreateInitialSetUpMessage,
            GetSetUpMessage,
            UpdateSetupMessage,
        ],
        verify_key: VerifyKey,
    ) -> Union[SuccessResponseMessage, GetSetUpResponse, UpdateSetupMessage]:
        return NodeSetupService.msg_handler_map[type(msg)](
            msg=msg, node=node, verify_key=verify_key
        )

    @staticmethod
    def message_handler_types() -> List[Type[ImmediateSyftMessageWithReply]]:
        return [
            CreateInitialSetUpMessage,
            GetSetUpMessage,
            UpdateSetupMessage,
        ]
