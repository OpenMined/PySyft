# stdlib
from typing import Any

# relative
from .....oblv.oblv_tensor_wrapper import OblvTensorWrapper
from ...abstract.node import AbstractNodeClient
from ...enums import ResponseObjectEnum
from ..action.exception_action import ExceptionMessage
from ..node_service.oblv.oblv_messages import CheckEnclaveConnectionMessage
from ..node_service.oblv.oblv_messages import CreateKeyPairMessage
from ..node_service.oblv.oblv_messages import GetPublicKeyMessage
from ..node_service.oblv.oblv_messages import PublishDatasetMessage
from .request_api import RequestAPI


class OblvAPI(RequestAPI):
    def __init__(self, client: AbstractNodeClient):
        super().__init__(
            client=client,
            response_key=ResponseObjectEnum.OBLV,
        )

    def get_key(self, **kwargs: Any) -> Any:
        response = self.perform_api_request(syft_msg=GetPublicKeyMessage, content=kwargs)
        if isinstance(response, ExceptionMessage):
            raise response.exception_type
        else:
            content = getattr(
                response, "response"
            )
            if content is None:
                raise Exception(f"{type(self)} has no response")
            return content

    def create_key(self, **kwargs: Any) -> Any:
        response = self.perform_api_request(syft_msg=CreateKeyPairMessage, content=kwargs)
        if isinstance(response, ExceptionMessage):
            raise response.exception_type
        else:
            content = getattr(
                response, "resp_msg"
            )
            if content is None:
                raise Exception(f"{type(self)} has no response")
            return content

    def check_connection(self, **kwargs: Any) -> Any:
        response = self.perform_api_request(syft_msg=CheckEnclaveConnectionMessage, content=kwargs)
        if isinstance(response, ExceptionMessage):
            raise response.exception_type
        else:
            content = getattr(
                response, "resp_msg"
            )
            if content is None:
                raise Exception(f"{type(self)} has no response")
            return content

    def publish_dataset(self, **kwargs: Any) -> Any:
        connection = self.client.routes[0].connection.base_url
        response = self.perform_api_request(syft_msg=PublishDatasetMessage, content={**kwargs,"host_or_ip": connection.host_or_ip, "protocol": connection.protocol,"port": connection.port})
        if isinstance(response, ExceptionMessage):
            raise response.exception_type
        else:
            dataset_id = getattr(
                response, "dataset_id"
            )
            client = getattr(
                response, "client"
            )
            return OblvTensorWrapper(id=dataset_id, deployment_id=kwargs["deployment_id"], oblv_client=client)

    def __getitem__(self) -> Any:
        return self.get()
