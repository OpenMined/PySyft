# stdlib
from typing import List

# syft absolute
from syft import flags

if flags.USE_NEW_SERVICE:
    # syft absolute
    from syft.core.node.common.node_service.user_manager.new_user_messages import (
        CreateUserMessage,
    )
    from syft.core.node.common.node_service.user_manager.new_user_messages import (
        DeleteUserMessage,
    )
    from syft.core.node.common.node_service.user_manager.new_user_messages import (
        GetUserMessage,
    )
    from syft.core.node.common.node_service.user_manager.new_user_messages import (
        GetUsersMessage,
    )
    from syft.core.node.common.node_service.user_manager.new_user_messages import (
        UpdateUserMessage,
    )
else:
    from syft.core.node.common.node_service.user_manager.user_manager_service import (
        GetUserMessage,
    )
    from syft.core.node.common.node_service.user_manager.user_manager_service import (
        GetUsersMessage,
    )
    from syft.core.node.common.node_service.user_manager.user_manager_service import (
        UpdateUserMessage,
    )
    from syft.core.node.common.node_service.user_manager.user_manager_service import (
        CreateUserMessage,
    )
    from syft.core.node.common.node_service.user_manager.user_manager_service import (
        DeleteUserMessage,
    )


# syft absolute
from syft.core.node.common.node_service.user_manager.user_manager_service import (
    GetCandidatesMessage,
)
from syft.core.node.common.node_service.user_manager.user_manager_service import (
    ProcessUserCandidateMessage,
)

# grid absolute
from grid.api.users.models import User
from grid.api.users.models import UserCandidate
from grid.api.users.models import UserCreate
from grid.api.users.models import UserPrivate
from grid.api.users.models import UserUpdate
from grid.utils import send_message_with_reply


def create_user(new_user: UserCreate, current_user: UserPrivate) -> str:
    reply = send_message_with_reply(
        signing_key=current_user.get_signing_key(),
        message_type=CreateUserMessage,
        **dict(new_user)
    )
    if flags.USE_NEW_SERVICE:
        reply = type("message", (object,), dict(reply))()
    return reply.resp_msg


def get_user_requests(current_user: UserPrivate) -> List[UserCandidate]:
    reply = send_message_with_reply(
        signing_key=current_user.get_signing_key(), message_type=GetCandidatesMessage
    )
    return [user.upcast() for user in reply.content]


def process_applicant_request(
    current_user: UserPrivate, candidate_id: int, status: str
) -> str:
    reply = send_message_with_reply(
        signing_key=current_user.get_signing_key(),
        message_type=ProcessUserCandidateMessage,
        candidate_id=candidate_id,
        status=status,
    )
    return reply.resp_msg


def get_all_users(current_user: UserPrivate) -> List[User]:
    result = send_message_with_reply(
        signing_key=current_user.get_signing_key(), message_type=GetUsersMessage
    )
    if not flags.USE_NEW_SERVICE:
        result = [user.upcast() for user in result.content]
    return result


def get_user(user_id: int, current_user: UserPrivate) -> User:
    result = send_message_with_reply(
        signing_key=current_user.get_signing_key(),
        message_type=GetUserMessage,
        user_id=user_id,
    )
    if not flags.USE_NEW_SERVICE:
        result = result.content.upcast()

    return result


def update_user(
    user_id: int, current_user: UserPrivate, updated_user: UserUpdate
) -> str:
    reply = send_message_with_reply(
        signing_key=current_user.get_signing_key(),
        message_type=UpdateUserMessage,
        user_id=user_id,
        **updated_user.dict(exclude_unset=True)
    )
    if flags.USE_NEW_SERVICE:
        reply = type("message", (object,), dict(reply))()
    return reply.resp_msg


def delete_user(user_id: int, current_user: UserPrivate) -> str:
    reply = send_message_with_reply(
        signing_key=current_user.get_signing_key(),
        message_type=DeleteUserMessage,
        user_id=user_id,
    )
    if flags.USE_NEW_SERVICE:
        reply = type("message", (object,), dict(reply))()
    # return reply.message - if the other one doesn't work try this one? ;)
    return reply.resp_msg
