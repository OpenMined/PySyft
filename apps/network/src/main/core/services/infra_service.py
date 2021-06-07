# stdlib
from datetime import datetime
import os
import secrets
from typing import List
from typing import Type
from typing import Union

# third party
from nacl.encoding import HexEncoder
from nacl.signing import SigningKey
from nacl.signing import VerifyKey
from syft import deserialize
from syft import serialize
from syft.core.common.message import ImmediateSyftMessageWithReply

# syft relative
from syft.core.node.abstract.node import AbstractNode
from syft.core.node.common.service.auth import service_auth
from syft.core.node.common.service.node_service import ImmediateNodeServiceWithReply
from syft.core.node.common.service.node_service import ImmediateNodeServiceWithoutReply
from syft.core.node.domain.client import DomainClient

# from syft.grid.client import connect
from syft.grid.client.client import connect
from syft.grid.client.grid_connection import GridHTTPConnection
from syft.grid.connections.http_connection import HTTPConnection
from syft.grid.messages.infra_messages import CreateWorkerMessage
from syft.grid.messages.infra_messages import CreateWorkerResponse
from syft.grid.messages.infra_messages import DeleteWorkerMessage
from syft.grid.messages.infra_messages import DeleteWorkerResponse
from syft.grid.messages.infra_messages import GetWorkerMessage
from syft.grid.messages.infra_messages import GetWorkerResponse
from syft.grid.messages.infra_messages import GetWorkersMessage
from syft.grid.messages.infra_messages import GetWorkersResponse
from syft.proto.core.io.address_pb2 import Address as Address_PB

# grid relative
from ...core.database.environment.environment import states
from ...core.infrastructure import AWS_Serverfull
from ...core.infrastructure import Config
from ...core.infrastructure import Provider
from ..database.utils import model_to_json
from ..exceptions import AuthorizationError
from ..exceptions import MissingRequestKeyError

# TODO: Modify existing routes or add new ones, to
# 1. allow admin to get all workers deployed by a specific user
# 2. allow admin to get all workers deployed by all users


def create_worker_msg(
    msg: CreateWorkerMessage, node: AbstractNode, verify_key: VerifyKey
) -> CreateWorkerResponse:
    try:
        _current_user_id = msg.content.get("current_user", None)
        instance_type = msg.content.get("instance_type", None)

        if instance_type is None:
            raise MissingRequestKeyError

        config = Config(
            app=Config(name="worker", count=1, id=len(node.environments.all()) + 1),
            apps=[Config(name="worker", count=1)],
            serverless=False,
            websockets=False,
            provider=os.environ["CLOUD_PROVIDER"],
            vpc=Config(
                region=os.environ["REGION"],
                instance_type=Config(InstanceType=instance_type),
            ),
        )

        deployment = None
        deployed = False

        if config.provider == "aws":
            deployment = AWS_Serverfull(config=config)
        elif config.provider == "azure":
            pass
        elif config.provider == "gcp":
            pass

        if deployment.validate():
            env_parameters = {
                "id": config.app.id,
                "state": states["creating"],
                "provider": config.provider,
                "region": config.vpc.region,
                "instance_type": config.vpc.instance_type.InstanceType,
            }
            new_env = node.environments.register(**env_parameters)
            node.environments.association(user_id=_current_user_id, env_id=new_env.id)

            deployed, output = deployment.deploy()  # Deploy

            if deployed:
                node.environments.set(
                    id=config.app.id,
                    created_at=datetime.now(),
                    state=states["success"],
                    address=output["instance_0_endpoint"]["value"][0],
                )
                # TODO: Modify this (@ionesio)
                # node.in_memory_client_registry[output.] = env_client
            else:
                node.environments.set(id=config.app.id, state=states["failed"])
                raise Exception("Worker creation failed!")

        final_msg = "Worker created successfully!"
        return CreateWorkerResponse(
            address=msg.reply_to, status_code=200, content={"msg": final_msg}
        )
    except Exception as e:
        return CreateWorkerResponse(
            address=msg.reply_to, status_code=500, content={"error": str(e)}
        )


def get_worker_msg(
    msg: GetWorkerMessage, node: AbstractNode, verify_key: VerifyKey
) -> GetWorkerResponse:
    try:

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
        is_admin = users.can_manage_infrastructure(user_id=_current_user_id)

        if (int(worker_id) in env_ids) or is_admin:

            _msg = model_to_json(node.environments.first(id=int(worker_id)))
        else:
            _msg = {}

        return GetWorkerResponse(address=msg.reply_to, status_code=200, content=_msg)
    except Exception as e:
        return GetWorkerResponse(
            address=msg.reply_to, status_code=500, content={"error": str(e)}
        )


def get_workers_msg(
    msg: GetWorkersMessage, node: AbstractNode, verify_key: VerifyKey
) -> GetWorkersResponse:
    try:
        _current_user_id = msg.content.get("current_user", None)
        include_all = msg.content.get("include_all", False)
        include_failed = msg.content.get("include_failed", False)
        include_destroyed = msg.content.get("include_destroyed", False)

        if not _current_user_id:
            _current_user_id = node.users.first(
                verify_key=verify_key.encode(encoder=HexEncoder).decode("utf-8")
            ).id

        envs = node.environments.get_environments(user=_current_user_id)

        workers = []
        for env in envs:
            _env = node.environments.first(id=env.id)
            if (
                include_all
                or (_env.state == states["success"])
                or (include_failed and _env.state == states["failed"])
                or (include_destroyed and _env.state == states["destroyed"])
            ):
                workers.append(model_to_json(_env))

        _msg = {"workers": workers}

        return GetWorkersResponse(address=msg.reply_to, status_code=200, content=_msg)
    except Exception as e:
        return GetWorkersResponse(
            address=msg.reply_to, status_code=False, content={"error": str(e)}
        )


def del_worker_msg(
    msg: DeleteWorkerMessage, node: AbstractNode, verify_key: VerifyKey
) -> DeleteWorkerResponse:
    try:
        # Get Payload Content
        worker_id = msg.content.get("worker_id", None)
        _current_user_id = msg.content.get("current_user", None)

        users = node.users
        if not _current_user_id:
            _current_user_id = users.first(
                verify_key=verify_key.encode(encoder=HexEncoder).decode("utf-8")
            ).id

        is_admin = users.can_manage_infrastructure(user_id=_current_user_id)

        envs = [
            int(env.id)
            for env in node.environments.get_environments(user=_current_user_id)
        ]
        created_by_current_user = int(worker_id) in envs

        # Owner / Admin
        if not is_admin and not created_by_current_user:
            raise AuthorizationError("You're not allowed to delete this worker!")

        env = node.environments.first(id=worker_id)
        _config = Config(provider=env.provider, app=Config(name="worker", id=worker_id))

        success = Provider(_config).destroy()
        if success:
            node.environments.set(
                id=worker_id, state=states["destroyed"], destroyed_at=datetime.now()
            )
        else:
            raise Exception("Worker deletion failed")

        return DeleteWorkerResponse(
            address=msg.reply_to,
            status_code=200,
            content={"msg": "Worker was deleted successfully!"},
        )
    except Exception as e:
        return DeleteWorkerResponse(
            address=msg.reply_to, status_code=False, content={"error": str(e)}
        )


# def update_worker_msg(
#     msg: UpdateWorkerMessage,
#     node: AbstractNode,
#     verify_key: VerifyKey,
# ) -> UpdateWorkerResponse:
#     # Get Payload Content
#     _worker_id = msg.content.get("worker_id", None)
#     _current_user_id = msg.content.get("current_user", None)
#
#     env_parameters = {
#         i: msg.content[i]
#         for i in msg.content.keys()
#         if i in list(Environment.__table__.columns.keys())
#     }
#
#     users = node.users
#     if not _current_user_id:
#         _current_user_id = users.first(
#             verify_key=verify_key.encode(encoder=HexEncoder).decode("utf-8")
#         ).id
#
#     _current_user = users.first(id=_current_user_id)
#
#     # Owner / Admin
#     if users.can_manage_infrastructure(user_id=_current_user_id):
#         node.environments.modify({"id": _worker_id}, env_parameters)
#     else:  # Env Owner
#         envs = [
#             int(env.id)
#             for env in node.environments.get_environments(user=_current_user_id)
#         ]
#         if int(_worker_id) in envs:
#             node.environments.modify({"id": _worker_id}, env_parameters)
#         else:
#             raise AuthorizationError(
#                 "You're not allowed to update this environment information!"
#             )
#     return UpdateWorkerResponse(
#         address=msg.reply_to,
#         status_code=200,
#         content={"msg": "Worker was updated succesfully!"},
#     )


class DomainInfrastructureService(ImmediateNodeServiceWithReply):

    msg_handler_map = {
        CreateWorkerMessage: create_worker_msg,
        # CheckWorkerDeploymentMessage: check_worker_deployment_msg,
        # UpdateWorkerMessage: update_worker_msg,
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
            GetWorkerMessage,
            GetWorkersMessage,
            DeleteWorkerMessage,
        ],
        verify_key: VerifyKey,
    ) -> Union[
        CreateWorkerResponse,
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
            # CheckWorkerDeploymentMessage,
            # UpdateWorkerMessage,
            GetWorkerMessage,
            GetWorkersMessage,
            DeleteWorkerMessage,
        ]
