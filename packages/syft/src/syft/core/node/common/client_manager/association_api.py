# stdlib
from typing import Any
from typing import Dict

# relative
from ...abstract.node import AbstractNodeClient
from ...enums import AssociationRequestResponses
from ...enums import RequestAPIFields
from ...enums import ResponseObjectEnum
from ...exceptions import PyGridClientException
from ..node_service.association_request.association_request_messages import (
    DeleteAssociationRequestMessage,
)
from ..node_service.association_request.association_request_messages import (
    GetAssociationRequestMessage,
)
from ..node_service.association_request.association_request_messages import (
    GetAssociationRequestsMessage,
)
from ..node_service.association_request.association_request_messages import (
    RespondAssociationRequestMessage,
)
from ..node_service.association_request.association_request_messages import (
    SendAssociationRequestMessage,
)
from ..node_service.success_resp_message import ErrorResponseMessage
from .request_api import RequestAPI


class AssociationRequestAPI(RequestAPI):
    def __init__(self, client: AbstractNodeClient):
        super().__init__(
            client=client,
            create_msg=SendAssociationRequestMessage,
            get_msg=GetAssociationRequestMessage,
            get_all_msg=GetAssociationRequestsMessage,
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

    def get(self, **kwargs: Any) -> Any:
        association_table = self.perform_api_request(
            syft_msg=self._get_message, content=kwargs
        )

        content = getattr(
            association_table, "content", getattr(association_table, "metadata", None)
        )
        if content is None:
            raise Exception(f"{type(self)} has no content or metadata field")

        obj_data = dict(content)
        obj_data[RequestAPIFields.SOURCE] = association_table.source
        obj_data[RequestAPIFields.TARGET] = association_table.target

        return self.to_obj(obj_data)

    def __getitem__(self, key: int) -> Any:
        return self.get(association_id=key)

    def __delitem__(self, key: int) -> None:
        self.delete(association_id=key)

    def to_obj(self, result: Dict[Any, Any]) -> Any:
        _association_obj = super().to_obj(result)

        _content = {
            RequestAPIFields.SOURCE: result[RequestAPIFields.SOURCE],
            RequestAPIFields.TARGET: result[RequestAPIFields.TARGET],
        }

        def _accept() -> Dict[str, str]:
            _content[RequestAPIFields.RESPONSE] = AssociationRequestResponses.ACCEPT
            return self.perform_api_request(
                syft_msg=RespondAssociationRequestMessage, content=_content
            )

        def _deny() -> Dict[str, str]:
            _content[RequestAPIFields.RESPONSE] = AssociationRequestResponses.DENY
            return self.perform_api_request(
                syft_msg=RespondAssociationRequestMessage, content=_content
            )

        _association_obj.accept = _accept
        _association_obj.deny = _deny

        return _association_obj
