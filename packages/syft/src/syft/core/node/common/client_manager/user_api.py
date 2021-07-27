# stdlib
from typing import Any
from typing import Type

# relative
from .....logger import logger
from ....node.common.node import Node
from ....node.domain.enums import ResponseObjectEnum
from ..node_service.user_manager.user_messages import CreateUserMessage
from ..node_service.user_manager.user_messages import DeleteUserMessage
from ..node_service.user_manager.user_messages import GetUserMessage
from ..node_service.user_manager.user_messages import GetUsersMessage
from ..node_service.user_manager.user_messages import UpdateUserMessage
from .request_api import RequestAPI


class UserRequestAPI(RequestAPI):
    def __init__(self, node: Type[Node]):
        super().__init__(
            node=node,
            create_msg=CreateUserMessage,
            get_msg=GetUserMessage,
            get_all_msg=GetUsersMessage,
            update_msg=UpdateUserMessage,
            delete_msg=DeleteUserMessage,
            response_key=ResponseObjectEnum.USER,
        )

    def __getitem__(self, key: int) -> Any:
        return self.get(user_id=key)

    def __delitem__(self, key: int) -> None:
        self.delete(user_id=key)

    def create(self, **kwargs: Any) -> None:
        try:
            response = self.perform_api_request(
                syft_msg=self._create_message, content=kwargs
            )
            logger.info(response.resp_msg)
        except Exception as e:
            for user in self.all():
                if user["email"] == kwargs["email"]:
                    print(
                        "Ignoring: user with email:" + user["email"] + " already exists"
                    )
                    return
            raise e
