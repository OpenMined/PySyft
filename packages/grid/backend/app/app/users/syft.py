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
from app.models.messages import GridToSyftMessage
from app.users.models import User
from app.users.models import UserCreate
from app.users.models import UserPrivate
from app.users.models import UserUpdate


def create_user(new_user: UserCreate, current_user: UserPrivate) -> str:
    message = GridToSyftMessage(
        signing_key=current_user.get_signing_key(),
        syft_message_type=CreateUserMessage,
    )
    response = message.send_with_reply(**dict(new_user))
    return response.resp_msg


def get_all_users(current_user: UserPrivate) -> List[User]:
    message = GridToSyftMessage(
        signing_key=current_user.get_signing_key(),
        syft_message_type=GetUsersMessage,
    )
    response = message.send_with_reply()
    return [user.upcast() for user in response.content]


def get_user(user_id: int, current_user: UserPrivate) -> User:
    message = GridToSyftMessage(
        signing_key=current_user.get_signing_key(), syft_message_type=GetUserMessage
    )
    response = message.send_with_reply(user_id=user_id)
    return response.content.upcast()


def update_user(
    user_id: int, current_user: UserPrivate, updated_user: UserUpdate
) -> str:
    message = GridToSyftMessage(
        signing_key=current_user.get_signing_key(),
        syft_message_type=UpdateUserMessage,
    )
    response = message.send_with_reply(user_id=user_id, **dict(updated_user))
    return response.resp_msg


def delete_user(user_id: int, current_user: UserPrivate) -> None:
    message = GridToSyftMessage(
        signing_key=current_user.get_signing_key(),
        syft_message_type=DeleteUserMessage,
    )
    message.send_with_reply(user_id=user_id)
