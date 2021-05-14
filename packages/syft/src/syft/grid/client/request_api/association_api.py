# stdlib
from typing import Any
from typing import Callable
from typing import Dict

# syft relative
from ...messages.association_messages import DeleteAssociationRequestMessage
from ...messages.association_messages import GetAssociationRequestMessage
from ...messages.association_messages import GetAssociationRequestsMessage
from ...messages.association_messages import RespondAssociationRequestMessage
from ...messages.association_messages import SendAssociationRequestMessage
from ..enums import AssociationRequestResponses
from ..enums import RequestAPIFields
from ..enums import ResponseObjectEnum
from ..exceptions import PyGridClientException
from .request_api import GridRequestAPI


class AssociationRequestAPI(GridRequestAPI):
    def __init__(self, send: Callable):
        super().__init__(
            create_msg=SendAssociationRequestMessage,
            get_msg=GetAssociationRequestMessage,
            get_all_msg=GetAssociationRequestsMessage,
            delete_msg=DeleteAssociationRequestMessage,
            send=send,
            response_key=ResponseObjectEnum.ASSOCIATION_REQUEST,
        )

    def update(self, **kwargs: Any) -> Dict[Any, Any]:  # type: ignore
        raise PyGridClientException(
            "You can not update an association request, try to send another one instead."
        )

    def __getitem__(self, key: int) -> Any:
        return self.get(association_request_id=key)

    def __delitem__(self, key: int) -> None:
        self.delete(association_request_id=key)

    def to_obj(self, result: Dict[Any, Any]) -> Any:
        _association_obj = super().to_obj(result)

        _content = {
            RequestAPIFields.ADDRESS: _association_obj.address,
            RequestAPIFields.HANDSHAKE: _association_obj.handshake_value,
            RequestAPIFields.SENDER_ADDRESS: _association_obj.sender_address,
        }

        def _accept() -> Dict[str, str]:
            _content[RequestAPIFields.VALUE] = AssociationRequestResponses.ACCEPT
            return self.send(
                grid_msg=RespondAssociationRequestMessage, content=_content
            )

        def _deny() -> Dict[str, str]:
            _content[RequestAPIFields.VALUE] = AssociationRequestResponses.DENY
            return self.send(
                grid_msg=RespondAssociationRequestMessage, content=_content
            )

        _association_obj.accept = _accept
        _association_obj.deny = _deny

        return _association_obj
