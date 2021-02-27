# stdlib
import secrets
from typing import List
from typing import Type
from typing import Union
from ..database.environment.environment import Environment

# third party
from nacl.signing import VerifyKey
from nacl.encoding import HexEncoder
from nacl.signing import SigningKey

# syft relative
from syft.core.node.abstract.node import AbstractNode
from syft.core.node.common.service.auth import service_auth
from syft.core.node.common.service.node_service import ImmediateNodeServiceWithReply
from syft.core.node.common.service.node_service import ImmediateNodeServiceWithoutReply
from syft.core.common.message import ImmediateSyftMessageWithReply

# from syft.grid.client import connect
from syft.grid.connections.http_connection import HTTPConnection
from syft.core.node.domain.client import DomainClient
from ..database.utils import model_to_json

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

from ..exceptions import AuthorizationError


def create_worker_msg(
    msg: CreateWorkerMessage,
    node: AbstractNode,
    verify_key: VerifyKey,
) -> CreateWorkerResponse:
    try:
        # TODO:
        # 1 - Deploy a Worker into the cloud using the parameters in msg.content
        # 2 - Save worker adress/metadata at node.workers

        _current_user_id = msg.content.get("current_user", None)

        users = node.users

        if not _current_user_id:
            _current_user_id = users.first(
                verify_key=verify_key.encode(encoder=HexEncoder).decode("utf-8")
            ).id

        _current_user = node.users.first(id=_current_user_id)
        env_parameters = {
            i: msg.content[i]
            for i in msg.content.keys()
            if i in list(Environment.__table__.columns.keys())
        }
        # env_client = connect(
        #    url=msg.content["address"],  # Domain Address
        #    conn_type=HTTPConnection,  # HTTP Connection Protocol
        #    client_type=DomainClient,
        #    user_key=SigningKey(
        #        _current_user.private_key.encode("utf-8"), encoder=HexEncoder
        #    ),
        # )

        # env_parameters["syft_address"] = (
        #    env_client.address.serialize().SerializeToString().decode("ISO-8859-1")
        # )
        new_env = node.environments.register(**env_parameters)
        node.environments.association(user_id=_current_user_id, env_id=new_env.id)

        # node.in_memory_client_registry[env_client.domain_id] = env_client

        final_msg = "Worker created succesfully!"
        return CreateWorkerResponse(
            address=msg.reply_to,
            status_code=200,
            content={"msg": final_msg},
        )
    except Exception as e:
        return CreateWorkerResponse(
            address=msg.reply_to,
            status_code=500,
            content={"error": str(e)},
        )


def check_worker_deployment_msg(
    msg: CheckWorkerDeploymentMessage,
    node: AbstractNode,
    verify_key: VerifyKey,
) -> CheckWorkerDeploymentResponse:
    try:
        # TODO:
        # Check Cloud Deployment progress
        # PS: msg.content is optional, but can be used to map different cloud deployment processes
        final_msg = {}  # All data about deployment progress.
        final_status = True

        return CheckWorkerDeploymentMessage(
            address=msg.reply_to,
            status_code=final_status,
            content={"deployment_status": final_msg},
        )
    except Exception as e:
        return CheckWorkerDeploymentMessage(
            address=msg.reply_to,
            status_code=500,
            content={"error": str(e)},
        )


def get_worker_msg(
    msg: GetWorkerMessage,
    node: AbstractNode,
    verify_key: VerifyKey,
) -> GetWorkerResponse:
    try:
        # TODO:
        # final_msg = node.workers[msg.content["worker_id"]]
        worker_id = msg.content.get("worker_id", None)
        _current_user_id = msg.content.get("current_user", None)

        users = node.users

        if not _current_user_id:
            _current_user_id = users.first(
                verify_key=verify_key.encode(encoder=HexEncoder).decode("utf-8")
            ).id
        env_ids = [
            env.id for env in node.environments.get_environments(user=_current_user_id)
        ]

        if int(worker_id) in env_ids:
            _msg = model_to_json(node.environments.first(id=int(worker_id)))
        else:
            _msg = {}

        return GetWorkerResponse(
            address=msg.reply_to,
            status_code=200,
            content=_msg,
        )
    except Exception as e:
        return GetWorkerResponse(
            address=msg.reply_to,
            status_code=500,
            content={"error": str(e)},
        )


def get_workers_msg(
    msg: GetWorkersMessage,
    node: AbstractNode,
    verify_key: VerifyKey,
) -> GetWorkersResponse:
    try:
        # TODO:
        # final_msg = node.workers
        try:
            _current_user_id = msg.content.get("current_user", None)
        except Exception:
            _current_user_id = None

        users = node.users

        if not _current_user_id:
            _current_user_id = users.first(
                verify_key=verify_key.encode(encoder=HexEncoder).decode("utf-8")
            ).id

        _current_user = node.users.first(id=_current_user_id)

        envs = node.environments.get_environments(user=_current_user_id)

        _msg = [model_to_json(node.environments.first(id=env.id)) for env in envs]

        return GetWorkersResponse(
            address=msg.reply_to,
            status_code=200,
            content=_msg,
        )
    except Exception as e:
        return GetWorkersResponse(
            address=msg.reply_to,
            status_code=False,
            content={"error": str(e)},
        )


def del_worker_msg(
    msg: DeleteWorkerMessage,
    node: AbstractNode,
    verify_key: VerifyKey,
) -> DeleteWorkerResponse:
    # Get Payload Content
    _worker_id = msg.content.get("worker_id", None)
    _current_user_id = msg.content.get("current_user", None)

    users = node.users
    if not _current_user_id:
        _current_user_id = users.first(
            verify_key=verify_key.encode(encoder=HexEncoder).decode("utf-8")
        ).id

    _current_user = users.first(id=_current_user_id)

    # Owner / Admin
    if users.can_manage_infrastructure(user_id=_current_user_id):
        node.environments.delete_associations(environment_id=_worker_id)
        node.environments.delete(id=_worker_id)
    else:  # Env Owner
        envs = [
            int(env.id)
            for env in node.environments.get_environments(user=_current_user_id)
        ]
        if int(_worker_id) in envs:
            node.environments.delete_associations(environment_id=_worker_id)
            node.environments.delete(id=_worker_id)
        else:
            raise AuthorizationError(
                "You're not allowed to delete this environment information!"
            )

    return DeleteWorkerResponse(
        address=msg.reply_to,
        status_code=200,
        content={"msg": "Worker was deleted succesfully!"},
    )


def update_worker_msg(
    msg: UpdateWorkerMessage,
    node: AbstractNode,
    verify_key: VerifyKey,
) -> UpdateWorkerResponse:
    # Get Payload Content
    _worker_id = msg.content.get("worker_id", None)
    _current_user_id = msg.content.get("current_user", None)

    env_parameters = {
        i: msg.content[i]
        for i in msg.content.keys()
        if i in list(Environment.__table__.columns.keys())
    }

    users = node.users
    if not _current_user_id:
        _current_user_id = users.first(
            verify_key=verify_key.encode(encoder=HexEncoder).decode("utf-8")
        ).id

    _current_user = users.first(id=_current_user_id)

    # Owner / Admin
    if users.can_manage_infrastructure(user_id=_current_user_id):
        node.environments.modify({"id": _worker_id}, env_parameters)
    else:  # Env Owner
        envs = [
            int(env.id)
            for env in node.environments.get_environments(user=_current_user_id)
        ]
        if int(_worker_id) in envs:
            node.environments.modify({"id": _worker_id}, env_parameters)
        else:
            raise AuthorizationError(
                "You're not allowed to update this environment information!"
            )
    return UpdateWorkerResponse(
        address=msg.reply_to,
        status_code=200,
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
        return DomainInfrastructureService.msg_handler_map[type(msg)](
            msg=msg, node=node, verify_key=verify_key
        )

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
