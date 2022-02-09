# stdlib
import logging
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Type

# third party
from pandas import DataFrame

# relative
# from ....common.message import SyftMessage
from ...abstract.node import AbstractNodeClient
from ...domain.enums import RequestAPIFields
from ..action.exception_action import ExceptionMessage
from ..node_service.generic_payload.messages import GenericPayloadMessageWithReply
from ..node_service.generic_payload.syft_message import SyftMessage


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
        self._create_message = create_msg
        self._get_message = get_msg
        self._get_all_message = get_all_msg
        self._update_message = update_msg
        self._delete_message = delete_msg
        self._response_key = response_key
        self.client = client
        self.perform_request = self.perform_api_request_generic

    def create(self, **kwargs: Any) -> None:
        response = self.perform_request(
            syft_msg=self._create_message, content=kwargs  # type: ignore
        )
        logging.info(response.message)

    def get(self, **kwargs: Any) -> Any:
        return self.to_obj(
            self.perform_request(syft_msg=self._get_message, content=kwargs)
        )

    def all(self) -> List[Any]:
        return sum(
            self.perform_request(syft_msg=self._get_all_message).dict().values(), []
        )

    def pandas(self) -> DataFrame:
        return DataFrame(self.all())

    def update(self, **kwargs: Any) -> None:
        response = self.perform_request(syft_msg=self._update_message, content=kwargs)
        logging.info(response.message)

    def delete(self, **kwargs: Any) -> None:
        response = self.perform_request(syft_msg=self._delete_message, content=kwargs)
        logging.info(response.message)

    def to_obj(self, result: Any) -> Any:
        if result:
            _class_name = self._response_key.capitalize()
            result = type(_class_name, (object,), result.dict())()

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

    def perform_api_request_generic(
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
        signed_msg = syft_msg_constructor(
            address=self.client.address, reply_to=self.client.address, kwargs=content
        ).sign(
            signing_key=self.client.signing_key
        )  # type: ignore
        response = self.client.send_immediate_msg_with_reply(msg=signed_msg)
        if isinstance(response, ExceptionMessage):
            raise response.exception_type
        else:
            return response.payload

    def _repr_html_(self) -> str:
        return self.pandas()._repr_html_()
