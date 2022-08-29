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

# relative
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
                    response = self.send_new_message_request(
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
    
    @property
    def send_new_message_request(self):
        self.__update_message_type_import()
        return self.perform_request
    
    def __update_message_type_import(self) -> None:
        # Auxiliar method used to exchange between Old and New User Messages in execution time.
        # NOTE: This auxiliar method is necessary only for User API and should be deleted after
        # Message refactory task.        
        if flags.USE_NEW_SERVICE:
            # relative
            from ..node_service.user_manager.new_user_messages import CreateUserMessage as CreateUserMessage
            from ..node_service.user_manager.new_user_messages import DeleteUserMessage as GetUserMessage
            from ..node_service.user_manager.new_user_messages import GetUserMessage as GetUsersMessage
            from ..node_service.user_manager.new_user_messages import GetUsersMessage as UpdateUserMessage
            from ..node_service.user_manager.new_user_messages import UpdateUserMessage as DeleteUserMessage
        else:
            # relative
            # type: ignore[override]
            from ..node_service.user_manager.user_messages import CreateUserMessage  # type: ignore
            from ..node_service.user_manager.user_messages import DeleteUserMessage  # type: ignore
            from ..node_service.user_manager.user_messages import GetUserMessage  # type: ignore
            from ..node_service.user_manager.user_messages import GetUsersMessage  # type: ignore
            from ..node_service.user_manager.user_messages import UpdateUserMessage  # type: ignore
        
        self._create_message=CreateUserMessage
        self._get_message=GetUserMessage
        self._get_all_message=GetUsersMessage
        self._update_message=UpdateUserMessage
        self._delete_message=DeleteUserMessage