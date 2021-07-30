# stdlib
from typing import List
from syft.core.node.common.node_service.object_request.object_request_messages import GetRequestMessage, UpdateRequestMessage

# syft absolute
from syft.core.node.common.node_service.object_request.object_request_service import (
    GetRequestsMessage,
)
# grid absolute
from app.requests.models import Request
from app.users.models import UserPrivate
from app.utils import send_message_with_reply


def get_all_requests(current_user: UserPrivate) -> List[Request]:
    reply = send_message_with_reply(
        signing_key=current_user.get_signing_key(), message_type=GetRequestsMessage
    )
    return [request.upcast() for request in reply.content] # upcast?


def get_request(current_user: UserPrivate, request_id: int) -> Request:
    reply = send_message_with_reply(signing_key=current_user.get_signing_key(), message_type=GetRequestMessage, request_id=request_id)
    return reply.upcast()

def update_request(current_user: UserPrivate, request_id: str, status: str) -> str:
    reply = send_message_with_reply(signing_key=current_user.get_signing_key(), message_type=UpdateRequestMessage, request_id=request_id, status=status)
    return reply.resp_msg

