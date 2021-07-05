# stdlib
import logging
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Type
from typing import Union

# third party
from pandas import DataFrame

# relative
from ....common.message import SyftMessage
from ....node.common.action.exception_action import ExceptionMessage
from ....node.common.node import Node
from ....node.domain.enums import RequestAPIFields


class RequestAPI:
    def __init__(
        self,
        node: Type[Node],
        create_msg: Optional[Type[SyftMessage]] = None,
        get_msg: Optional[Type[SyftMessage]] = None,
        get_all_msg: Optional[Type[SyftMessage]] = None,
        update_msg: Optional[Type[SyftMessage]] = None,
        delete_msg: Optional[Type[SyftMessage]] = None,
        response_key: str = "",
    ):
        self.__create_message = create_msg
        self.__get_message = get_msg
        self.__get_all_message = get_all_msg
        self.__update_message = update_msg
        self.__delete_message = delete_msg
        self.__response_key = response_key
        self.node = node

    def create(self, **kwargs: Any) -> None:
        response = self.perform_api_request(
            syft_msg=self.__create_message, content=kwargs
        )
        logging.info(response.resp_msg)

    def get(self, **kwargs: Any) -> Any:
        return self.to_obj(
            self.perform_api_request(
                syft_msg=self.__get_message, content=kwargs
            ).content.upcast()
        )

    def all(self, pandas: bool = False) -> Union[DataFrame, Dict[Any, Any]]:
        result = [
            content.upcast()
            for content in self.perform_api_request(
                syft_msg=self.__get_all_message
            ).content
        ]
        if pandas:
            result = DataFrame(result)

        return result

    def update(self, **kwargs: Any) -> None:
        response = self.perform_api_request(
            syft_msg=self.__update_message, content=kwargs
        )
        logging.info(response.resp_msg)

    def delete(self, **kwargs: Any) -> None:
        response = self.perform_api_request(
            syft_msg=self.__delete_message, content=kwargs
        )
        logging.info(response.resp_msg)

    def to_obj(self, result: Any) -> Any:
        if result:
            _class_name = self.__response_key.capitalize()
            result = type(_class_name, (object,), result)()

        return result

    def perform_api_request(
        self,
        syft_msg: Type[SyftMessage],
        content: Optional[Dict[Any, Any]] = None,
    ) -> Dict[Any, Any]:
        if content is None:
            content = {}
        content[RequestAPIFields.ADDRESS] = self.node.address
        content[RequestAPIFields.REPLY_TO] = self.node.address
        signed_msg = syft_msg(**content).sign(signing_key=self.node.signing_key)
        response = self.node.send_immediate_msg_with_reply(msg=signed_msg)
        if isinstance(response, ExceptionMessage):
            raise response.exception_type
        else:
            return response
