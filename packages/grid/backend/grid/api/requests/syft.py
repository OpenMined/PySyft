# stdlib
from typing import List

# syft absolute
from syft.core.node.common.node_service.object_request.object_request_messages import (
    GetRequestMessage,
)
from syft.core.node.common.node_service.object_request.object_request_messages import (
    UpdateRequestMessage,
)
from syft.core.node.common.node_service.object_request.object_request_service import (
    GetBudgetRequestsMessage,
)
from syft.core.node.common.node_service.object_request.object_request_service import (
    GetRequestsMessage,
)

# grid absolute
from grid.api.users.models import UserPrivate
from grid.utils import send_message_with_reply

# relative
from .models import BudgetRequestResponse
from .models import Request
from .models import RequestUpdate


def get_all_requests(current_user: UserPrivate) -> List[Request]:
    reply = send_message_with_reply(
        signing_key=current_user.get_signing_key(), message_type=GetRequestsMessage
    )
    return [request for request in reply.content]  # upcast?


def get_all_budget_requests(current_user: UserPrivate) -> List[BudgetRequestResponse]:
    reply = send_message_with_reply(
        signing_key=current_user.get_signing_key(),
        message_type=GetBudgetRequestsMessage,
    )
    return [request for request in reply.content]  # upcast?


def get_request(current_user: UserPrivate, request_id: str) -> Request:
    reply = send_message_with_reply(
        signing_key=current_user.get_signing_key(),
        message_type=GetRequestMessage,
        request_id=request_id,
    )
    return reply


def update_request(
    current_user: UserPrivate, request_id: str, updated_request: RequestUpdate
) -> str:
    reply = send_message_with_reply(
        signing_key=current_user.get_signing_key(),
        message_type=UpdateRequestMessage,
        request_id=request_id,
        status=updated_request.status,
    )
    return reply.status
