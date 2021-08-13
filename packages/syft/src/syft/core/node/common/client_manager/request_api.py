# stdlib
import logging
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Type

# third party
from pandas import DataFrame

# syft absolute
from syft.core.node.abstract.node import AbstractNodeClient

# relative
from ....common.message import SyftMessage
from ....node.common.action.exception_action import ExceptionMessage
from ....node.domain.enums import RequestAPIFields


class RequestAPI:
    def __init__(
        self,
        client: AbstractNodeClient,
        create_msg: Optional[Type[SyftMessage]] = None,
        get_msg: Optional[Type[SyftMessage]] = None,
        get_all_msg: Optional[Type[SyftMessage]] = None,
        update_msg: Optional[Type[SyftMessage]] = None,
        delete_msg: Optional[Type[SyftMessage]] = None,
        response_key: str = "",
    ):
        self.__create_message = create_msg
        self._create_message = self.__create_message
        self.__get_message = get_msg
        self.__get_all_message = get_all_msg
        self._get_all_message = get_all_msg
        self.__update_message = update_msg
        self.__delete_message = delete_msg
        self.__response_key = response_key
        self.client = client

    def create(self, **kwargs: Any) -> None:
        response = self.perform_api_request(
            syft_msg=self.__create_message, content=kwargs
        )
        logging.info(response.resp_msg)

    def get(self, **kwargs: Any) -> Any:
        association_table = self.perform_api_request(
            syft_msg=self.__get_message, content=kwargs
        )
        obj_data = dict(association_table.metadata)
        obj_data[RequestAPIFields.SOURCE] = association_table.source
        obj_data[RequestAPIFields.TARGET] = association_table.target

        return self.to_obj(obj_data)

    def all(self) -> List[Any]:
        result = [
            content
            for content in self.perform_api_request(
                syft_msg=self.__get_all_message
            ).metadatas
        ]

        return result

    def pandas(self) -> DataFrame:
        return DataFrame(self.all())

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
        syft_msg: Optional[Type[SyftMessage]],
        content: Optional[Dict[Any, Any]] = None,
    ) -> Any:
        if syft_msg is None:
            raise ValueError(
                "Can't perform this type of api request, the message is None."
            )
        else:
            syft_msg_constructor = syft_msg

        if content is None:
            content = {}
        content[RequestAPIFields.ADDRESS] = self.client.address
        content[RequestAPIFields.REPLY_TO] = self.client.address
        signed_msg = syft_msg_constructor(**content).sign(
            signing_key=self.client.signing_key
        )  # type: ignore
        response = self.client.send_immediate_msg_with_reply(msg=signed_msg)
        if isinstance(response, ExceptionMessage):
            raise response.exception_type
        else:
            return response

    def _repr_html_(self) -> str:
        return self.pandas()._repr_html_()
