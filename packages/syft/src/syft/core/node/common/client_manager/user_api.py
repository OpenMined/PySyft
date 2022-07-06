# stdlib
from typing import Any
from typing import Dict

# relative
from .....experimental_flags import flags
from .....logger import logger
from ...abstract.node import AbstractNodeClient
from ...enums import ResponseObjectEnum
from ..exceptions import AuthorizationError
from ..node_service.user_auth.user_auth_messages import UserLoginMessageWithReply

if flags.USE_NEW_SERVICE:
    # relative
    from ..node_service.user_manager.new_user_messages import CreateUserMessage
    from ..node_service.user_manager.new_user_messages import DeleteUserMessage
    from ..node_service.user_manager.new_user_messages import GetUserMessage
    from ..node_service.user_manager.new_user_messages import GetUsersMessage
    from ..node_service.user_manager.new_user_messages import UpdateUserMessage
else:
    # relative
    # type: ignore[override]
    from ..node_service.user_manager.user_messages import CreateUserMessage  # type: ignore
    from ..node_service.user_manager.user_messages import DeleteUserMessage  # type: ignore
    from ..node_service.user_manager.user_messages import GetUserMessage  # type: ignore
    from ..node_service.user_manager.user_messages import GetUsersMessage  # type: ignore
    from ..node_service.user_manager.user_messages import UpdateUserMessage  # type: ignore

# relative
from .request_api import RequestAPI


class UserRequestAPI(RequestAPI):
    def __init__(self, client: AbstractNodeClient):
        super().__init__(
            client=client,
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
            if "pdf" in kwargs.keys():
                response = self.client.routes[0].connection.send_files(  # type: ignore
                    "/users",
                    kwargs.get("pdf"),
                    form_name="new_user",
                    form_values=kwargs,  # type: ignore
                )  # type: ignore
                logger.info(response)
            else:
                response_message = ""
                if flags.USE_NEW_SERVICE:
                    response = self.perform_request(
                        syft_msg=self._create_message, content=kwargs  # type: ignore
                    )
                    response_message = response.payload.message
                    logger.info(response.payload.message)
                else:
                    response = self.perform_api_request(
                        syft_msg=self._create_message, content=kwargs
                    )
                    response_message = response.resp_msg
                    logger.info(response.resp_msg)
                print(response_message)

        except Exception as e:
            print("failing to create user", e)
            try:
                for user in self.all():
                    if user["email"] == kwargs["email"]:
                        print(
                            "Ignoring: user with email:"
                            + user["email"]
                            + " already exists"
                        )
                        return
            except AuthorizationError as exc:
                print("No permission to check users", exc)

            raise e

    def login(self, email: str, password: str) -> Dict[str, Any]:
        response = self.perform_api_request_generic(
            syft_msg=UserLoginMessageWithReply,
            content={"email": email, "password": password},
        )

        return response.payload.kwargs.upcast()  # type: ignore
