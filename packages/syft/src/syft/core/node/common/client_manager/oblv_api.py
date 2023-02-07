# stdlib
from typing import Any
from typing import Dict
from typing import Optional
from typing import Type
from typing import Union

# relative
from .....core.pointer.pointer import Pointer
from .....oblv.deployment_client import DeploymentClient
from .....oblv.oblv_enclave_pointer import OblvEnclavePointer
from .....telemetry import instrument
from ....common.message import SyftMessage  # type: ignore
from ...abstract.node import AbstractNodeClient
from ...enums import RequestAPIFields
from ...enums import ResponseObjectEnum
from ..action.exception_action import ExceptionMessage
from ..node_service.oblv.oblv_messages import CheckEnclaveConnectionMessage
from ..node_service.oblv.oblv_messages import CreateKeyPairMessage
from ..node_service.oblv.oblv_messages import DeductBudgetMessage
from ..node_service.oblv.oblv_messages import GetPublicKeyMessage
from ..node_service.oblv.oblv_messages import PublishApprovalMessage
from ..node_service.oblv.oblv_messages import TransferDatasetMessage
from .request_api import RequestAPI


@instrument
class OblvAPI(RequestAPI):
    def __init__(self, client: AbstractNodeClient):
        super().__init__(
            client=client,
            response_key=ResponseObjectEnum.OBLV,
        )

    def get_key(self, **kwargs: Any) -> Any:
        response = self.perform_api_request(
            syft_msg=GetPublicKeyMessage, content=kwargs
        )
        if isinstance(response, ExceptionMessage):
            raise response.exception_type
        else:
            content = getattr(response, "response")
            if content is None:
                raise Exception(f"{type(self)} has no response")
            return content

    def create_key(self, **kwargs: Any) -> Any:
        response = self.perform_api_request(
            syft_msg=CreateKeyPairMessage, content=kwargs
        )
        if isinstance(response, ExceptionMessage):
            raise response.exception_type
        else:
            content = getattr(response, "resp_msg")
            if content is None:
                raise Exception(f"{type(self)} has no response")
            return content

    def check_connection(self, deployment: DeploymentClient, **kwargs: Any) -> Any:
        content = {
            "deployment_id": deployment.deployment_id,
            "oblv_client": deployment.oblv_client,
        }
        response = self.perform_api_request(
            syft_msg=CheckEnclaveConnectionMessage, content=content
        )
        if isinstance(response, ExceptionMessage):
            raise response.exception_type
        else:
            content = getattr(response, "resp_msg")
            if content is None:
                raise Exception(f"{type(self)} has no response")
            return content

    def transfer_dataset(
        self,
        deployment: DeploymentClient,
        dataset: Union[str, Pointer],
        **kwargs: Any,
    ) -> Any:
        content = {
            "deployment_id": deployment.deployment_id,
            "oblv_client": deployment.oblv_client,
        }
        dataset_id = (
            dataset if isinstance(dataset, str) else dataset.id_at_location.to_string()
        )
        content.update({"dataset_id": dataset_id})
        response = self.perform_api_request(
            syft_msg=TransferDatasetMessage, content=content
        )
        if isinstance(response, ExceptionMessage):
            raise response.exception_type
        else:
            dataset_id = getattr(response, "dataset_id")
            return OblvEnclavePointer(id=dataset_id, deployment_client=deployment)

    def publish_budget(
        self, deployment_id, publish_request_id, client, **kwargs: Any
    ) -> Any:
        response = self.oblv_perform_api_request_without_reply(
            syft_msg=PublishApprovalMessage,
            content={
                "result_id": publish_request_id,
                "oblv_client": client,
                "deployment_id": deployment_id,
            },
        )
        if isinstance(response, ExceptionMessage):
            raise response.exception_type

    def publish_request_budget_deduction(
        self, deployment_id, publish_request_id, budget_to_deduct, client, **kwargs: Any
    ) -> Any:
        response = self.oblv_perform_api_request_without_reply(
            syft_msg=DeductBudgetMessage,
            content={
                "result_id": publish_request_id,
                "oblv_client": client,
                "budget_to_deduct": budget_to_deduct,
                "deployment_id": deployment_id,
            },
        )
        if isinstance(response, ExceptionMessage):
            raise response.exception_type

    def __getitem__(self) -> Any:
        return self.get()

    def oblv_perform_api_request_without_reply(
        self,
        syft_msg: Type[SyftMessage],
        content: Optional[Dict[Any, Any]] = None,
        timeout: Optional[int] = None,
    ) -> Any:
        if syft_msg is None:
            raise ValueError(
                "Can't perform this type of api request, the message is None."
            )
        else:
            syft_msg_constructor = syft_msg

        if content is None:
            content = {}
        content[RequestAPIFields.ADDRESS] = self.client.node_uid

        signed_msg = syft_msg_constructor(**content).sign(
            signing_key=self.client.signing_key
        )  # type: ignore
        response = self.client.send_immediate_msg_without_reply(
            msg=signed_msg, timeout=timeout
        )
        if isinstance(response, ExceptionMessage):
            raise response.exception_type
        else:
            return response
