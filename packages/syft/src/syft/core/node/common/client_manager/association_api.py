# stdlib
from typing import Any
from typing import Dict
from typing import List

# relative
from ...abstract.node import AbstractNodeClient
from ...enums import AssociationRequestResponses
from ...enums import RequestAPIFields
from ...enums import ResponseObjectEnum
from ...exceptions import PyGridClientException
from ..node_service.association_request.association_request_messages import (
    DeleteAssociationRequestMessage,
)

from ..node_service.association_request.new_association_request import (
    GetAllAssociationsMessage,
)
from ..node_service.association_request.new_association_request import (
    GetAssociationRequestMessage
)
from ..node_service.association_request.new_association_request import (
    TriggerAssociationRequestMessage,
)
from ..node_service.association_request.new_association_request import (
    ProcessAssociationRequestMessage,
)
from ..node_service.success_resp_message import ErrorResponseMessage
from .request_api import RequestAPI


class AssociationRequestAPI(RequestAPI):
    def __init__(self, client: AbstractNodeClient):
        super().__init__(
            client=client,
            create_msg=TriggerAssociationRequestMessage,
            get_msg=GetAssociationRequestMessage,
            get_all_msg=GetAllAssociationsMessage,
            delete_msg=DeleteAssociationRequestMessage,
            response_key=ResponseObjectEnum.ASSOCIATION_REQUEST,
        )

    def create(self, **kwargs: Any) -> None:
        retry = kwargs.pop("retry")
        response = self.perform_api_request(  # type: ignore
            syft_msg=self._create_message, content=kwargs
        )

        if isinstance(response, ErrorResponseMessage):
            if retry > 0:
                print(response.resp_msg, " Retrying", retry, " ...")
                kwargs["retry"] = retry - 1
                self.create(**kwargs)
            else:
                raise Exception(response.resp_msg)

    def update(self, **kwargs: Any) -> Dict[Any, Any]:  # type: ignore
        raise PyGridClientException(
            "You can not update an association request, try to send another one instead."
        )

    def create(self, **kwargs: Any) -> Any:
        self.perform_request(syft_msg=self._create_message, content=kwargs)  # type: ignore

    def get(self, **kwargs: Any) -> Any:
        content = self.perform_request(syft_msg=self._get_message, content=kwargs).kwargs["association_request"]

        #association_table = self.perform_api_request(
        #    syft_msg=self._get_message, content=kwargs
        #)

        #content = getattr(
        #    association_table, "content", getattr(association_table, "metadata", None)
        #)
        
        if content is None:
            raise Exception(f"{type(self)} has no content or metadata field")
        
        return self.to_obj(dict(content))

    def all(self) -> List[Any]:
        result = self.perform_request(syft_msg=self._get_all_message).kwargs[
            "association_requests"
        ]
        return result

    def __getitem__(self, key: int) -> Any:
        return self.get(association_id=key)

    def __delitem__(self, key: int) -> None:
        self.delete(association_id=key)

    def to_obj(self, result: Dict[Any, Any]) -> Any:
        _association_obj = super().to_obj(result)

        _content = {}
        _content[RequestAPIFields.NODE_ADDRESS.value] = _association_obj.node_address
        def _accept() -> Dict[str, str]:
            _content[RequestAPIFields.ACCEPT.value] = True
            return self.perform_request(
                syft_msg=ProcessAssociationRequestMessage, content=_content
            )

        def _deny() -> Dict[str, str]:
            _content[RequestAPIFields.ACCEPT.value] = False
            return self.perform_request(
                syft_msg=ProcessAssociationRequestMessage, content=_content
            )

        _association_obj.accept = _accept
        _association_obj.deny = _deny

        return _association_obj
