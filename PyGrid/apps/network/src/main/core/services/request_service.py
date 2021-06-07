# stdlib
from typing import List
from typing import Type
from typing import Union

# third party
from nacl.encoding import HexEncoder
from nacl.signing import VerifyKey
from syft.core.common.message import ImmediateSyftMessageWithReply

# syft relative
from syft.core.node.abstract.node import AbstractNode
from syft.core.node.common.service.auth import service_auth
from syft.core.node.common.service.node_service import ImmediateNodeServiceWithReply
from syft.core.node.common.service.node_service import ImmediateNodeServiceWithoutReply
from syft.grid.messages.request_messages import CreateRequestMessage
from syft.grid.messages.request_messages import CreateRequestResponse
from syft.grid.messages.request_messages import DeleteRequestMessage
from syft.grid.messages.request_messages import DeleteRequestResponse
from syft.grid.messages.request_messages import GetRequestMessage
from syft.grid.messages.request_messages import GetRequestResponse
from syft.grid.messages.request_messages import GetRequestsMessage
from syft.grid.messages.request_messages import GetRequestsResponse
from syft.grid.messages.request_messages import UpdateRequestMessage
from syft.grid.messages.request_messages import UpdateRequestResponse

# grid relative
from ..database.utils import model_to_json
from ..exceptions import AuthorizationError
from ..exceptions import InvalidParameterValueError
from ..exceptions import MissingRequestKeyError


def create_request_msg(
    msg: CreateRequestMessage,
    node: AbstractNode,
    verify_key: VerifyKey,
) -> CreateRequestResponse:
    # Get Payload Content
    current_user_id = msg.content.get("current_user", None)
    object_id = msg.content.get("object_id", None)
    reason = msg.content.get("reason", None)
    request_type = msg.content.get("request_type", None)

    users = node.users

    if not current_user_id:
        current_user = users.first(
            verify_key=verify_key.encode(encoder=HexEncoder).decode("utf-8")
        )
    else:
        current_user = users.first(id=current_user_id)

    # Check if object_id/reason/request_type fields are empty
    missing_paramaters = not object_id or not reason or not request_type
    if missing_paramaters:
        raise MissingRequestKeyError(
            message="Invalid request payload, empty fields (object_id/reason/request_type)!"
        )

    valid_paramaters = request_type == "permissions" or request_type == "budget"

    if not valid_paramaters:
        raise InvalidParameterValueError(
            message='Request type should be either "permissions” or “budget”.'
        )

    requests = node.data_requests
    request_obj = requests.create_request(
        user_id=current_user.id,
        user_name=current_user.email,
        object_id=object_id,
        reason=reason,
        request_type=request_type,
    )
    request_json = model_to_json(request_obj)

    return CreateRequestResponse(
        address=msg.reply_to,
        status_code=200,
        content=request_json,
    )


def get_request_msg(
    msg: GetRequestMessage,
    node: AbstractNode,
    verify_key: VerifyKey,
) -> GetRequestResponse:

    # Get Payload Content
    request_id = msg.content.get("request_id", None)
    current_user_id = msg.content.get("current_user", None)

    users = node.users

    if not current_user_id:
        current_user_id = users.first(
            verify_key=verify_key.encode(encoder=HexEncoder).decode("utf-8")
        ).id

    requests = node.data_requests
    request = requests.first(id=request_id)

    # A user can get a request if he's the owner of that request
    # or has the can_triage_requests permission
    allowed = request.user_id == current_user_id or users.can_triage_requests(
        user_id=current_user_id
    )

    if allowed:
        request_json = model_to_json(request)
    else:
        raise AuthorizationError("You're not allowed to get Request information!")

    return GetRequestResponse(
        address=msg.reply_to,
        status_code=200,
        content=request_json,
    )


def get_all_request_msg(
    msg: GetRequestsMessage,
    node: AbstractNode,
    verify_key: VerifyKey,
) -> GetRequestsResponse:

    # Get Payload Content
    current_user_id = msg.content.get("current_user", None)

    users = node.users

    if not current_user_id:
        current_user_id = users.first(
            verify_key=verify_key.encode(encoder=HexEncoder).decode("utf-8")
        ).id

    allowed = users.can_triage_requests(user_id=current_user_id)

    if allowed:
        requests = node.data_requests
        requests = requests.all()
        requests_json = [model_to_json(requests) for requests in requests]
    else:
        raise AuthorizationError("You're not allowed to get Request information!")

    return GetRequestsResponse(
        address=msg.reply_to,
        status_code=200,
        content=requests_json,
    )


def update_request_msg(
    msg: DeleteRequestMessage,
    node: AbstractNode,
    verify_key: VerifyKey,
) -> DeleteRequestResponse:

    # Get Payload Content
    request_id = msg.content.get("request_id", None)
    status = msg.content.get("status", None)
    current_user_id = msg.content.get("current_user", None)

    users = node.users

    if not current_user_id:
        current_user_id = users.first(
            verify_key=verify_key.encode(encoder=HexEncoder).decode("utf-8")
        ).id

    # Check if status field is empty
    missing_paramaters = not status
    if missing_paramaters:
        raise MissingRequestKeyError(
            message="Invalid request payload, empty fields (status)!"
        )

    allowed = users.can_triage_requests(user_id=current_user_id)

    if allowed:
        requests = node.data_requests

        if status not in ["accepted", "denied"]:
            raise InvalidParameterValueError(
                message='Request status should be either "accepted" or "denied"'
            )

        if status == "accepted":
            request = requests.first(id=request_id)
            object_id = request.object_id

            # Accessing and updating the datase metadata
            storage = node.disk_store
            read_permission = {
                "verify_key": verify_key.encode(encoder=HexEncoder).decode("utf-8"),
                "request_id": request_id,
            }
            storage.update_dataset_metadata(
                key=object_id, read_permissions=read_permission
            )

        # TODO:
        # 1 - The logic to change a user privacy budget needs to be implemented,
        # as soon as this logic is ready this need to be updated.

        requests.set(request_id=request_id, status=status)
    else:
        raise AuthorizationError("You're not allowed to update Request information!")

    return DeleteRequestResponse(
        address=msg.reply_to,
        status_code=200,
        content={"msg": "Request Updated!"},
    )


def del_request_msg(
    msg: DeleteRequestMessage,
    node: AbstractNode,
    verify_key: VerifyKey,
) -> DeleteRequestResponse:

    # Get Payload Content
    request_id = msg.content.get("request_id", None)
    current_user_id = msg.content.get("current_user", None)

    users = node.users

    if not current_user_id:
        current_user_id = users.first(
            verify_key=verify_key.encode(encoder=HexEncoder).decode("utf-8")
        ).id

    requests = node.data_requests
    request = requests.first(id=request_id)

    # Only the creator of a request may delete their request.
    if request.user_id == current_user_id:
        requests.delete(id=request_id)
    else:
        raise AuthorizationError("You're not allowed to delete this Request!")

    return DeleteRequestResponse(
        address=msg.reply_to,
        status_code=200,
        content={"msg": "Request deleted!"},
    )


class RequestService(ImmediateNodeServiceWithReply):

    msg_handler_map = {
        CreateRequestMessage: create_request_msg,
        GetRequestMessage: get_request_msg,
        GetRequestsMessage: get_all_request_msg,
        UpdateRequestMessage: update_request_msg,
        DeleteRequestMessage: del_request_msg,
    }

    @staticmethod
    @service_auth(guests_welcome=True)
    def process(
        node: AbstractNode,
        msg: Union[
            CreateRequestMessage,
            GetRequestMessage,
            GetRequestsMessage,
            UpdateRequestMessage,
            DeleteRequestMessage,
        ],
        verify_key: VerifyKey,
    ) -> Union[
        CreateRequestResponse,
        GetRequestResponse,
        GetRequestsResponse,
        UpdateRequestResponse,
        DeleteRequestResponse,
    ]:

        return RequestService.msg_handler_map[type(msg)](
            msg=msg, node=node, verify_key=verify_key
        )

    @staticmethod
    def message_handler_types() -> List[Type[ImmediateSyftMessageWithReply]]:
        return [
            CreateRequestMessage,
            GetRequestMessage,
            GetRequestsMessage,
            GetRequestsMessage,
            UpdateRequestMessage,
            DeleteRequestMessage,
        ]
