# stdlib
from typing import Any

# relative
from ...abstract.node import AbstractNodeClient
from ...enums import ResponseObjectEnum
from ..node_service.oblv.oblv_messages import CreateKeyPairMessage
from ..node_service.oblv.oblv_messages import GetPublicKeyMessage
from .request_api import RequestAPI


class OblvAPI(RequestAPI):
    def __init__(self, client: AbstractNodeClient):
        super().__init__(
            client=client,
            create_msg=CreateKeyPairMessage,
            get_msg=GetPublicKeyMessage,
            response_key=ResponseObjectEnum.OBLV,
        )

    def get(self, **kwargs: Any) -> Any:
        response = self.perform_api_request(syft_msg=self._get_message, content=kwargs)
        content = getattr(
            response, "response"
        )
        if content is None:
            raise Exception(f"{type(self)} has no response")
        return content
    
    
    def __getitem__(self) -> Any:
        return self.get()
    
    
