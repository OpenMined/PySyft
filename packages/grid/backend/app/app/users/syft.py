# stdlib
from typing import List

# syft absolute
from syft.core.node.common.node_service.user_manager.user_manager_service import (
    CreateUserMessage,
)
from syft.core.node.common.node_service.user_manager.user_manager_service import (
    DeleteUserMessage,
)
from syft.core.node.common.node_service.user_manager.user_manager_service import (
    GetUserMessage,
)
from syft.core.node.common.node_service.user_manager.user_manager_service import (
    GetUsersMessage,
)
from syft.core.node.common.node_service.user_manager.user_manager_service import (
    UpdateUserMessage,
)

# grid absolute
from app.users.models import User
from app.users.models import UserCreate
from app.users.models import UserPrivate
from app.users.models import UserUpdate
from app.utils import send_message_with_reply


def create_user(new_user: UserCreate, current_user: UserPrivate) -> str:
    reply = send_message_with_reply(
        signing_key=current_user.get_signing_key(),
        message_type=CreateUserMessage,
        **dict(new_user)
    )
    return reply.resp_msg


def get_all_users(current_user: UserPrivate) -> List[User]:
    reply = send_message_with_reply(
        signing_key=current_user.get_signing_key(), message_type=GetUsersMessage
    )
    return [user.upcast() for user in reply.content]


def get_user(user_id: int, current_user: UserPrivate) -> User:
    reply = send_message_with_reply(
        signing_key=current_user.get_signing_key(),
        message_type=GetUserMessage,
        user_id=user_id,
    )
    return reply.content.upcast()


def update_user(
    user_id: int, current_user: UserPrivate, updated_user: UserUpdate
) -> str:
    reply = send_message_with_reply(
        signing_key=current_user.get_signing_key(),
        message_type=UpdateUserMessage,
        user_id=user_id,
        **dict(updated_user)
    )
    return reply.resp_msg


def delete_user(user_id: int, current_user: UserPrivate) -> str:
    reply = send_message_with_reply(
        signing_key=current_user.get_signing_key(),
        message_type=DeleteUserMessage,
        user_id=user_id,
    )
    return reply.resp_msg
