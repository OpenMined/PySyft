# stdlib
from typing import Any
from typing import Dict

# third party
from pandas import DataFrame

# syft relative
from ...messages.association_messages import DeleteAssociationRequestMessage
from ...messages.association_messages import GetAssociationRequestMessage
from ...messages.association_messages import GetAssociationRequestsMessage
from ...messages.association_messages import RespondAssociationRequestMessage
from ...messages.association_messages import SendAssociationRequestMessage
from .request_api import GridRequestAPI


class AssociationRequestAPI(GridRequestAPI):
    response_key = "association-request"

    def __init__(self, send):
        super().__init__(
            create_msg=SendAssociationRequestMessage,
            get_msg=GetAssociationRequestMessage,
            get_all_msg=GetAssociationRequestsMessage,
            update_msg=None,
            delete_msg=DeleteAssociationRequestMessage,
            send=send,
            response_key=AssociationRequestAPI.response_key,
        )

    def update(self, **kwargs):
        raise NotImplementedError("Method not implemented to Association Requests!")

    def __getitem__(self, key):
        return self.get(association_request_id=key)

    def __delitem__(self, key):
        self.delete(association_request_id=key)

    def to_obj(self, result: Dict[Any, Any]):
        _association_obj = super().to_obj(result)

        _content = {
            "address": _association_obj.address,
            "handshake": _association_obj.handshake_value,
            "sender_address": _association_obj.sender_address,
        }

        def _accept():
            _content["value"] = "accept"
            return self.send_method(
                grid_msg=RespondAssociationRequestMessage, content=_content
            )

        def _deny():
            _content["value"] = "deny"
            return self.send_method(
                grid_msg=RespondAssociationRequestMessage, content=_content
            )

        _association_obj.accept = _accept
        _association_obj.deny = _deny

        return _association_obj
