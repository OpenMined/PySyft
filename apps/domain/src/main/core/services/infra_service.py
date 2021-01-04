# stdlib
import secrets
from typing import List
from typing import Type
from typing import Union

# third party
from nacl.signing import VerifyKey

# syft relative
from syft.core.node.abstract.node import AbstractNode
from syft.core.node.common.service.auth import service_auth
from syft.core.node.common.service.node_service import ImmediateNodeServiceWithReply
from syft.core.node.common.service.node_service import ImmediateNodeServiceWithoutReply
from syft.decorators.syft_decorator_impl import syft_decorator
from syft.core.common.message import ImmediateSyftMessageWithReply

from syft.grid.messages.infra_messages import (
    CreateWorkerMessage,
    CreateWorkerResponse,
    CheckWorkerDeploymentMessage,
    CheckWorkerDeploymentResponse,
    GetWorkerResponse,
    GetWorkerMessage,
    GetWorkerResponse,
    GetWorkersMessage,
    GetWorkersResponse,
    UpdateWorkerMessage,
    UpdateWorkerResponse,
    DeleteWorkerMessage,
    DeleteWorkerResponse,
)


@syft_decorator(typechecking=True)
def create_worker_msg(
    msg: CreateWorkerMessage,
) -> CreateWorkerResponse:
    return CreateWorkerResponse(
        address=msg.reply_to,
        success=True,
        content={"msg": "Worker created succesfully!"},
    )


@syft_decorator(typechecking=True)
def check_worker_deployment_msg(
    msg: CheckWorkerDeploymentMessage,
) -> CheckWorkerDeploymentResponse:
    return CheckWorkerDeploymentMessage(
        address=msg.reply_to,
        success=True,
        content={"msg": "Association request received!"},
    )


@syft_decorator(typechecking=True)
def get_worker_msg(
    msg: GetWorkerMessage,
) -> GetWorkerResponse:
    return GetWorkerResponse(
        address=msg.reply_to,
        success=True,
        content={
            "worker": {"id": "9846165", "address": "159.156.128.165", "datasets": 25320}
        },
    )


@syft_decorator(typechecking=True)
def get_workers_msg(
    msg: GetWorkersMessage,
) -> GetWorkersResponse:
    return GetWorkersResponse(
        address=msg.reply_to,
        success=True,
        content={
            "workers": [
                {"id": "546513231a", "address": "159.156.128.165", "datasets": 25320},
                {"id": "asfa16f5aa", "address": "138.142.125.125", "datasets": 2530},
                {"id": "af61ea3a3f", "address": "19.16.98.146", "datasets": 2320},
                {"id": "af4a51adas", "address": "15.59.18.165", "datasets": 5320},
            ]
        },
    )


@syft_decorator(typechecking=True)
def del_worker_msg(
    msg: DeleteWorkerMessage,
) -> DeleteWorkerResponse:
    return DeleteWorkerResponse(
        address=msg.reply_to,
        success=True,
        content={"msg": "Worker was deleted succesfully!"},
    )


@syft_decorator(typechecking=True)
def update_worker_msg(
    msg: UpdateWorkerMessage,
) -> UpdateWorkerResponse:
    return UpdateWorkerResponse(
        address=msg.reply_to,
        success=True,
        content={"msg": "Worker was updated succesfully!"},
    )


class DomainInfrastructureService(ImmediateNodeServiceWithReply):

    msg_handler_map = {
        CreateWorkerMessage: create_worker_msg,
        CheckWorkerDeploymentMessage: check_worker_deployment_msg,
        UpdateWorkerMessage: update_worker_msg,
        GetWorkerMessage: get_worker_msg,
        GetWorkersMessage: get_workers_msg,
        DeleteWorkerMessage: del_worker_msg,
    }

    @staticmethod
    @service_auth(guests_welcome=True)
    def process(
        node: AbstractNode,
        msg: Union[
            CreateWorkerMessage,
            CheckWorkerDeploymentMessage,
            UpdateWorkerMessage,
            GetWorkerMessage,
            GetWorkersMessage,
            DeleteWorkerMessage,
        ],
        verify_key: VerifyKey,
    ) -> Union[
        CreateWorkerResponse,
        CheckWorkerDeploymentResponse,
        UpdateWorkerResponse,
        GetWorkerResponse,
        GetWorkersResponse,
        DeleteWorkerResponse,
    ]:
        return DomainInfrastructureService.msg_handler_map[type(msg)](msg=msg)

    @staticmethod
    def message_handler_types() -> List[Type[ImmediateSyftMessageWithReply]]:
        return [
            CreateWorkerMessage,
            CheckWorkerDeploymentMessage,
            UpdateWorkerMessage,
            GetWorkerMessage,
            GetWorkersMessage,
            DeleteWorkerMessage,
        ]
