# stdlib
from typing import Any
from typing import Dict

# syft absolute
from syft.core.node.abstract.node import AbstractNodeClient
from syft.core.node.common.node_service.association_request.association_request_messages import (
    DeleteAssociationRequestMessage,
)
from syft.core.node.common.node_service.association_request.association_request_messages import (
    GetAssociationRequestMessage,
)
from syft.core.node.common.node_service.association_request.association_request_messages import (
    GetAssociationRequestsMessage,
)
from syft.core.node.common.node_service.association_request.association_request_messages import (
    RespondAssociationRequestMessage,
)
from syft.core.node.common.node_service.association_request.association_request_messages import (
    SendAssociationRequestMessage,
)
from syft.core.node.domain.exceptions import PyGridClientException

# relative
from ....node.domain.enums import AssociationRequestResponses
from ....node.domain.enums import RequestAPIFields
from ....node.domain.enums import ResponseObjectEnum
from ...common.client_manager.request_api import RequestAPI


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

    def update(self, **kwargs: Any) -> Dict[Any, Any]:  # type: ignore
        raise PyGridClientException(
            "You can not update an association request, try to send another one instead."
        )

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
