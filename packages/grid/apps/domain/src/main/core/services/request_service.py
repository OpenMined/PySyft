# stdlib
import time
from typing import List
from typing import Optional
from typing import Type
from typing import Union

# third party
from nacl.encoding import HexEncoder
from nacl.signing import VerifyKey
from syft.core.common.message import ImmediateSyftMessageWithReply
from syft.core.common.message import ImmediateSyftMessageWithoutReply
from syft.core.common.uid import UID

# syft relative
from syft.core.node.abstract.node import AbstractNode
from syft.core.node.common.node import DuplicateRequestException
from syft.core.node.common.service.auth import service_auth
from syft.core.node.common.service.node_service import ImmediateNodeServiceWithReply
from syft.core.node.common.service.node_service import ImmediateNodeServiceWithoutReply
from syft.core.node.domain.service.accept_or_deny_request_service import (
    AcceptOrDenyRequestMessage,
)
from syft.core.node.domain.service.get_all_requests_service import (
    GetAllRequestsResponseMessage,
)
from syft.core.node.domain.service.get_all_requests_service import GetAllRequestsMessage
from syft.core.node.domain.service.request_answer_message import RequestAnswerMessage
from syft.core.node.domain.service.request_answer_message import RequestAnswerResponse
from syft.core.node.domain.service.request_handler_service import (
    GetAllRequestHandlersMessage,
)
from syft.core.node.domain.service.request_handler_service import (
    GetAllRequestHandlersResponseMessage,
)
from syft.core.node.domain.service.request_handler_service import (
    UpdateRequestHandlerMessage,
)
from syft.core.node.domain.service.request_message import RequestMessage
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
from syft.util import validate_type

# grid relative
from ..database.utils import model_to_json
from ..datasets.dataset_ops import update_dataset_metadata
from ..exceptions import AuthorizationError
from ..exceptions import InvalidParameterValueError
from ..exceptions import MissingRequestKeyError
from ..exceptions import RequestError


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
    object_type = msg.content.get("object_type", "storable object")

    users = node.users

    if not current_user_id:
        current_user = users.first(
            verify_key=verify_key.encode(encoder=HexEncoder).decode("utf-8")
        )
    else:
        current_user = users.first(id=current_user_id)

    # since we reject/accept requests based on the ID, we don't want there to be
    # multiple requests with the same ID because this could cause security problems.
    _duplicate_request = node.data_requests.contain(
        object_id=object_id,
        verify_key=verify_key.encode(encoder=HexEncoder).decode("utf-8"),
    )

    if _duplicate_request:
        raise DuplicateRequestException(
            "You have already requested {}".format(msg.content["object_id"])
        )

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
    object_uid = UID.from_string(object_id)

    request_obj = requests.create_request(
        user_id=current_user.id,
        user_name=current_user.email,
        object_id=object_id,
        reason=reason,
        request_type=request_type,
        verify_key=verify_key.encode(encoder=HexEncoder).decode("utf-8"),
        object_type=object_type,
        tags=node.store[object_uid]._tags,
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

    _req = node.data_requests.first(id=request_id)

    if not _req:
        raise RequestError

    if status not in ["accepted", "denied"]:
        raise InvalidParameterValueError(
            message='Request status should be either "accepted" or "denied"'
        )

    _can_triage_request = node.users.can_triage_requests(user_id=current_user_id)
    _current_user_key = verify_key.encode(encoder=HexEncoder).decode("utf-8")
    _req_owner = _current_user_key == _req.verify_key

    if status == "accepted" and _can_triage_request:
        tmp_obj = node.store[UID.from_string(_req.object_id)]
        tmp_obj.read_permissions[
            VerifyKey(_req.verify_key.encode("utf-8"), encoder=HexEncoder)
        ] = _req.id
        node.store[UID.from_string(_req.object_id)] = tmp_obj
        node.data_requests.set(request_id=_req.id, status=status)
    elif status == "denied" and (_can_triage_request or _req_owner):
        node.data_requests.set(request_id=_req.id, status=status)
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


# PySyft Services


def request_answer_msg(
    msg: RequestAnswerMessage, node: AbstractNode, verify_key: VerifyKey
):
    if verify_key is None:
        raise ValueError(
            "Can't process Request service without a given " "verification key"
        )

    status = node.data_requests.status(id=str(msg.request_id.value))  # type: ignore
    address = msg.reply_to
    return RequestAnswerResponse(
        request_id=msg.request_id, address=address, status=status
    )


# TODO: Check if this method/message should really be a service_without_reply message
def get_all_requests(
    msg: GetAllRequestsMessage, node: AbstractNode, verify_key: VerifyKey
):
    if verify_key is None:
        raise ValueError(
            "Can't process Request service without a given " "verification key"
        )

    current_user = node.users.first(
        verify_key=verify_key.encode(encoder=HexEncoder).decode("utf-8")
    )
    _can_triage_request = node.users.can_triage_requests(user_id=current_user.id)

    _requests = node.data_requests.all()

    if _can_triage_request:
        _requests = node.data_requests.all()
    else:
        _requests = node.data_requests.query(
            verify_key=verify_key.encode(encoder=HexEncoder).decode("utf-8")
        )

    data_requests = [
        RequestMessage(
            request_id=UID.from_string(req.id),
            request_description=req.reason,
            address=node.address,
            owner_address=node.address,
            object_id=UID.from_string(req.object_id),
            object_type=req.object_type,
            object_tags=req.tags,
            requester_verify_key=VerifyKey(
                req.verify_key.encode("utf-8"), encoder=HexEncoder
            ),
            timeout_secs=None,
        )
        for req in _requests
    ]

    return GetAllRequestsResponseMessage(requests=data_requests, address=msg.reply_to)


def get_all_request_handlers(
    msg: GetAllRequestHandlersMessage, node: AbstractNode, verify_key: VerifyKey
):

    if verify_key is None:
        raise ValueError(
            "Can't process Request service without a given " "verification key"
        )

    handlers = []
    if verify_key == node.root_verify_key:
        existing_handlers = getattr(node, "request_handlers", None)
        if existing_handlers is not None:
            handlers = existing_handlers

    return GetAllRequestHandlersResponseMessage(handlers=handlers, address=msg.reply_to)


class RequestService(ImmediateNodeServiceWithReply):

    msg_handler_map = {
        CreateRequestMessage: create_request_msg,
        GetRequestMessage: get_request_msg,
        GetRequestsMessage: get_all_request_msg,
        UpdateRequestMessage: update_request_msg,
        DeleteRequestMessage: del_request_msg,
        RequestAnswerMessage: request_answer_msg,
        GetAllRequestsMessage: get_all_requests,
        GetAllRequestHandlersMessage: get_all_request_handlers,
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
            RequestAnswerMessage,
            GetAllRequestsMessage,
            GetAllRequestHandlersMessage,
        ],
        verify_key: VerifyKey,
    ) -> Union[
        CreateRequestResponse,
        GetRequestResponse,
        GetRequestsResponse,
        UpdateRequestResponse,
        DeleteRequestResponse,
        RequestAnswerResponse,
        GetAllRequestsResponseMessage,
        GetAllRequestHandlersResponseMessage,
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
            RequestAnswerMessage,
            GetAllRequestsMessage,
            GetAllRequestHandlersMessage,
        ]


def build_request_message(
    msg: RequestMessage, node: AbstractNode, verify_key: VerifyKey
):
    if verify_key is None:
        raise ValueError(
            "Can't process Request service without a given " "verification key"
        )

    if msg.requester_verify_key != verify_key:
        raise Exception(
            "You tried to request access for a key that is not yours!"
            "You cannot do this! Whatever key you want to request access"
            "for must be the verify key that also verifies the message"
            "containing the request."
        )

    # since we reject/accept requests based on the ID, we don't want there to be
    # multiple requests with the same ID because this could cause security problems.
    _duplicate_request = node.data_requests.contain(
        object_id=str(msg.object_id.value),
        verify_key=verify_key.encode(encoder=HexEncoder).decode("utf-8"),
    )

    if _duplicate_request:
        raise DuplicateRequestException(
            f"You have already requested {msg.object_id} ",
            node.data_requests.all(),
            "My Requests",
        )

    current_user = node.users.first(
        verify_key=verify_key.encode(encoder=HexEncoder).decode("utf-8")
    )

    node.data_requests.create_request(
        user_id=current_user.id,
        user_name=current_user.email,
        object_id=str(msg.object_id.value),
        reason=msg.request_description,
        request_type="permissions",
        verify_key=verify_key.encode(encoder=HexEncoder).decode("utf-8"),
        object_type=msg.object_type,
        tags=node.store[msg.object_id]._tags,
    )


def accept_or_deny_request(
    msg: AcceptOrDenyRequestMessage, node: AbstractNode, verify_key: VerifyKey
):
    if verify_key is None:
        raise ValueError(
            "Can't process AcceptOrDenyRequestService without a specified verification key"
        )

    _msg: AcceptOrDenyRequestMessage = validate_type(msg, AcceptOrDenyRequestMessage)

    current_user = node.users.first(
        verify_key=verify_key.encode(encoder=HexEncoder).decode("utf-8")
    )

    _req = node.data_requests.first(id=str(_msg.request_id.value))
    _can_triage_request = node.users.can_triage_requests(user_id=current_user.id)
    if _msg.accept:
        if _req and _can_triage_request:
            tmp_obj = node.store[UID.from_string(_req.object_id)]
            tmp_obj.read_permissions[
                VerifyKey(_req.verify_key.encode("utf-8"), encoder=HexEncoder)
            ] = _req.id
            node.store[UID.from_string(_req.object_id)] = tmp_obj
            node.data_requests.set(request_id=_req.id, status="accepted")
    else:
        _req_owner = current_user.verify_key == _req.verify_key
        if _req and (_can_triage_request or _req_owner):
            node.data_requests.set(request_id=_req.id, status="denied")


def update_req_handler(
    node: AbstractNode,
    msg: UpdateRequestHandlerMessage,
    verify_key: Optional[VerifyKey] = None,
) -> None:
    if verify_key is None:
        raise ValueError(
            "Can't process Request service without a given " "verification key"
        )

    if verify_key == node.root_verify_key:
        replacement_handlers = []

        # find if there exists a handler match the handler passed in
        existing_handlers = getattr(node, "request_handlers", None)
        # debug(f"> Updating Request Handlers with existing: {existing_handlers}")
        if existing_handlers is not None:
            matched = None
            for existing_handler in existing_handlers:
                # we match two handlers according to their tags
                if existing_handler["tags"] == msg.handler["tags"]:
                    matched = existing_handler
                    # if an existing_handler match the passed in handler,
                    # we ignore it in for loop
                    continue
                else:
                    # if an existing_handler does not match the passed in
                    # handler, we keep it
                    replacement_handlers.append(existing_handler)

            if msg.keep:
                msg.handler["created_time"] = time.time()
                replacement_handlers.append(msg.handler)
                if matched is not None:
                    print(
                        f"> Replacing a Request Handler {matched} with: {msg.handler}"
                    )
                    # debug(
                    #    f"> Replacing a Request Handler {matched} with: {msg.handler}"
                    # )
                else:
                    print(f"> Adding a Request Handler {msg.handler}")
                    # debug(f"> Adding a Request Handler {msg.handler}")
            else:
                print(f"> Removing a Request Handler with: {msg.handler}")
                # debug(f"> Removing a Request Handler with: {msg.handler}")

            setattr(node, "request_handlers", replacement_handlers)
            # debug(f"> Finished Updating Request Handlers with: {existing_handlers}")
        else:
            # TODO: Replace line below
            print(f"> Node has no Request Handlers attribute: {type(node)}")
            # error(f"> Node has no Request Handlers attribute: {type(node)}")

    return


class RequestServiceWithoutReply(ImmediateNodeServiceWithoutReply):

    msg_handler_map = {
        RequestMessage: build_request_message,
        AcceptOrDenyRequestMessage: accept_or_deny_request,
        UpdateRequestHandlerMessage: update_req_handler,
    }

    @staticmethod
    @service_auth(guests_welcome=True)
    def process(
        node: AbstractNode,
        msg: Union[
            RequestMessage,
            AcceptOrDenyRequestMessage,
            UpdateRequestHandlerMessage,
        ],
        verify_key: VerifyKey,
    ) -> None:
        RequestServiceWithoutReply.msg_handler_map[type(msg)](
            msg=msg, node=node, verify_key=verify_key
        )

    @staticmethod
    def message_handler_types() -> List[Type[ImmediateSyftMessageWithoutReply]]:
        return [RequestMessage, AcceptOrDenyRequestMessage, UpdateRequestHandlerMessage]
