# stdlib
from typing import List

# syft absolute
from syft.core.node.common.node_service.object_request.new_object_request_messages import (
    NewGetBudgetRequestsMessage,
)
from syft.core.node.common.node_service.object_request.new_object_request_messages import (
    NewGetDataRequestsMessage,
)
from syft.core.node.common.node_service.object_request.new_object_request_messages import (
    NewGetRequestMessage,
)
from syft.core.node.common.node_service.object_request.new_object_request_messages import (
    NewUpdateRequestsMessage,
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
        signing_key=current_user.get_signing_key(),
        message_type=NewGetDataRequestsMessage,
    )
    return reply.requests  # upcast?


def get_all_budget_requests(current_user: UserPrivate) -> List[BudgetRequestResponse]:
    reply = send_message_with_reply(
        signing_key=current_user.get_signing_key(),
        message_type=NewGetBudgetRequestsMessage,
    )
    return reply.requests  # upcast?


def get_request(current_user: UserPrivate, request_id: str) -> Request:
    reply = send_message_with_reply(
        signing_key=current_user.get_signing_key(),
        message_type=NewGetRequestMessage,
        request_id=request_id,
    )
    return reply


def update_request(
    current_user: UserPrivate, request_id: str, updated_request: RequestUpdate
) -> str:
    reply = send_message_with_reply(
        signing_key=current_user.get_signing_key(),
        message_type=NewUpdateRequestsMessage,
        request_id=request_id,
        status=updated_request.status,
    )
    return reply.status
