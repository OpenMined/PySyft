# TODO: move this service back to PyGrid where it belongs

# # stdlib
# from datetime import datetime
# import os
# from typing import List
# from typing import Type
# from typing import Union
#
# # third party
# from nacl.encoding import HexEncoder
# from nacl.signing import VerifyKey
#
# # syft absolute
# from syft import serialize
# from syft.core.common.message import ImmediateSyftMessageWithReply
# from syft.core.node.abstract.node import AbstractNode
# from syft.core.node.common.service.auth import service_auth
# from syft.core.node.common.service.node_service import ImmediateNodeServiceWithReply
# from syft.grid.client.client import connect
# from syft.grid.client.grid_connection import GridHTTPConnection
# from .infra_manager_messages import CreateWorkerMessage
# from .infra_manager_messages import CreateWorkerResponse
# from .infra_manager_messages import DeleteWorkerMessage
# from .infra_manager_messages import DeleteWorkerResponse
# from .infra_manager_messages import GetWorkerInstanceTypesMessage
# from .infra_manager_messages import GetWorkerInstanceTypesResponse
# from .infra_manager_messages import GetWorkerMessage
# from .infra_manager_messages import GetWorkerResponse
# from .infra_manager_messages import GetWorkersMessage
# from .infra_manager_messages import GetWorkersResponse
#
# # relative
# from ....core.infrastructure import AWS_Serverfull
# from ....core.infrastructure import AZURE
# from ....core.infrastructure import Config
# from ....core.infrastructure import GCP
# from ....core.infrastructure import Provider
# from ....core.infrastructure import aws_utils
# from ....core.infrastructure import azure_utils
# from ....core.infrastructure import gcp_utils
# from ...database.tables.environment import states
# from ...database.utils import model_to_json
# from ...exceptions import AuthorizationError
# from ...exceptions import MissingRequestKeyError
#
# # TODO: Modify existing routes or add new ones, to
# # 1. allow admin to get all workers deployed by a specific user
# # 2. allow admin to get all workers deployed by all users
#
# SUPPORTED_PROVIDERS = [
#     "aws",
#     "azure",
#     "gcp",
# ]  # todo: add azure and gcp after testing worker deployment
# PROVIDER_UTILS = {
#     "aws": aws_utils,
#     "azure": azure_utils,
#     "gcp": gcp_utils,
# }
#
#
# def get_worker_instance_types_msg(
#     msg: GetWorkerInstanceTypesMessage, node: AbstractNode, verify_key: VerifyKey
# ) -> GetWorkerInstanceTypesResponse:
#     try:
#         _current_user_id = msg.content.get("current_user", None)
#         provider = os.environ.get("CLOUD_PROVIDER")
#         region = os.environ.get("REGION")
#
#         if provider not in SUPPORTED_PROVIDERS:
#             raise Exception("Provider not supported")
#
#         # todo: We can make worker deployment a permissible operation
#         # Users can deploy certain instance types (such as "Free Tier Instances") without permission
#         # But to deploy other instance types, example those which are costly, they would need to ask permission
#         # This servide would then return only those instance types which the users has the permission to deploy
#
#         if provider == "aws":
#             _msg = PROVIDER_UTILS[provider].get_all_instance_types(region)
#         elif provider == "gcp":
#             zone = os.environ.get("ZONE", None)
#             _msg = PROVIDER_UTILS[provider].get_all_instance_types(zone=zone)
#         elif provider == "azure":
#             location = os.environ.get("location", None)
#             _msg = PROVIDER_UTILS[provider].get_all_instance_types(location=location)
#         return GetWorkerInstanceTypesResponse(
#             address=msg.reply_to, status_code=200, content=_msg
#         )
#     except Exception as e:
#         return GetWorkerInstanceTypesResponse(
#             address=msg.reply_to, status_code=500, content={"error": str(e)}
#         )
#
#
# def create_worker_msg(
#     msg: CreateWorkerMessage, node: AbstractNode, verify_key: VerifyKey
# ) -> CreateWorkerResponse:
#     try:
#         _current_user_id = msg.content.get("current_user", None)
#         instance_type = msg.content.get("instance_type", None)
#         _worker_port = msg.content.get("port", 5001)
#
#         users = node.users
#
#         if not _current_user_id:
#             _current_user_id = users.first(
#                 verify_key=verify_key.encode(encoder=HexEncoder).decode("utf-8")
#             ).id
#
#         if instance_type is None:
#             raise MissingRequestKeyError
#
#         config = Config(
#             app=Config(name="worker", count=1, id=len(node.environments.all()) + 1),
#             apps=[Config(name="worker", count=1, port=_worker_port)],
#             serverless=False,
#             websockets=False,
#             provider=os.getenv("CLOUD_PROVIDER", "AWS"),
#             ##TODO(amr): encapsulate each cloud provider to config to aws, azure, gcp
#             vpc=Config(
#                 region=os.getenv("REGION", None),
#                 instance_type=Config(InstanceType=instance_type),
#             ),
#             azure=Config(
#                 location=os.getenv("location", None),
#                 subscription_id=os.getenv("subscription_id", None),
#                 client_id=os.getenv("client_id", None),
#                 client_secret=os.getenv("client_secret", None),
#                 tenant_id=os.getenv("tenant_id", None),
#                 vm_size=instance_type,
#             ),
#             gcp=Config(
#                 project_id=os.getenv("project_id", None),
#                 region=os.getenv("region", None),
#                 zone=os.getenv("zone", None),
#                 machine_type=instance_type,
#             ),
#         )
#
#         deployment = None
#         deployed = False
#
#         if config.provider == "aws":
#             deployment = AWS_Serverfull(config=config)
#         elif config.provider == "azure":
#             deployment = AZURE(config=config)
#         elif config.provider == "gcp":
#             deployment = GCP(config=config)
#
#         if deployment.validate():
#             deployed, output = deployment.deploy()  # Deploy
#             if deployed:
#                 env_parameters = {
#                     "id": config.app.id,
#                     "provider": config.provider,
#                     "created_at": datetime.now(),
#                     "state": states["success"],
#                     "address": output["instance_0_endpoint"]["value"][0]
#                     + ":"
#                     + str(_worker_port),
#                 }
#
#                 if config.provider == "aws":
#                     env_parameters["region"] = config.vpc.region
#                     env_parameters[
#                         "instance_type"
#                     ] = config.vpc.instance_type.InstanceType
#                 elif config.provider == "azure":
#                     env_parameters["region"] = config.azure.location
#                     env_parameters["instance_type"] = config.azure.vm_size
#                 elif config.provider == "gcp":
#                     env_parameters["region"] = config.gcp.region
#                     env_parameters["instance_type"] = config.gcp.machine_type
#
#                 new_env = node.environments.register(**env_parameters)
#                 node.environments.association(
#                     user_id=_current_user_id, env_id=new_env.id
#                 )
#             else:
#                 node.environments.set(id=config.app.id, state=states["failed"])
#                 raise Exception("Worker creation failed!")
#         final_msg = "Worker created successfully!"
#         return CreateWorkerResponse(
#             address=msg.reply_to, status_code=200, content={"message": final_msg}
#         )
#     except Exception as e:
#         return CreateWorkerResponse(
#             address=msg.reply_to, status_code=500, content={"error": str(e)}
#         )
#
#
# def get_worker_msg(
#     msg: GetWorkerMessage, node: AbstractNode, verify_key: VerifyKey
# ) -> GetWorkerResponse:
#     try:
#
#         worker_id = msg.content.get("worker_id", None)
#         _current_user_id = msg.content.get("current_user", None)
#
#         users = node.users
#
#         if not _current_user_id:
#             _current_user_id = users.first(
#                 verify_key=verify_key.encode(encoder=HexEncoder).decode("utf-8")
#             ).id
#
#         env_ids = [
#             env.id for env in node.environments.get_environments(user=_current_user_id)
#         ]
#         is_admin = users.can_manage_infrastructure(user_id=_current_user_id)
#
#         if (int(worker_id) in env_ids) or is_admin:
#             worker = node.environments.first(id=int(worker_id))
#
#             try:
#                 worker_client = connect(
#                     url="http://" + worker.address,
#                     conn_type=GridHTTPConnection,  # HTTP Connection Protocol
#                 )
#
#                 node.environments.set(
#                     id=worker.id,
#                     syft_address=serialize(worker_client.address)
#                     .SerializeToString()
#                     .decode("ISO-8859-1"),
#                 )
#
#                 node.in_memory_client_registry[worker_client.domain_id] = worker_client
#             except Exception as e:
#                 return GetWorkerResponse(
#                     address=msg.reply_to,
#                     status_code=500,
#                     content={"error": str(e)},
#                 )
#             _msg = model_to_json(node.environments.first(id=int(worker_id)))
#         else:
#             _msg = {}
#
#         return GetWorkerResponse(address=msg.reply_to, status_code=200, content=_msg)
#     except Exception as e:
#         return GetWorkerResponse(
#             address=msg.reply_to, status_code=500, content={"error": str(e)}
#         )
#
#
# def get_workers_msg(
#     msg: GetWorkersMessage, node: AbstractNode, verify_key: VerifyKey
# ) -> GetWorkersResponse:
#     try:
#         _current_user_id = msg.content.get("current_user", None)
#         include_all = msg.content.get("include_all", True)
#         include_failed = msg.content.get("include_failed", False)
#         include_destroyed = msg.content.get("include_destroyed", False)
#
#         if not _current_user_id:
#             _current_user_id = node.users.first(
#                 verify_key=verify_key.encode(encoder=HexEncoder).decode("utf-8")
#             ).id
#
#         envs = node.environments.get_environments(user=_current_user_id)
#
#         workers = []
#         print("Node environments: ", node.environments.all()[0].id)
#         for env in envs:
#             print("Here!", env.id)
#             _env = node.environments.first(id=env.id)
#
#             if (
#                 include_all
#                 or (_env.state == states["success"])
#                 or (include_failed and _env.state == states["failed"])
#                 or (include_destroyed and _env.state == states["destroyed"])
#             ):
#                 worker = model_to_json(_env)
#                 del worker["syft_address"]
#                 workers.append(worker)
#
#         _msg = workers
#
#         return GetWorkersResponse(address=msg.reply_to, status_code=200, content=_msg)
#     except Exception as e:
#         return GetWorkersResponse(
#             address=msg.reply_to, status_code=False, content={"error": str(e)}
#         )
#
#
# def del_worker_msg(
#     msg: DeleteWorkerMessage, node: AbstractNode, verify_key: VerifyKey
# ) -> DeleteWorkerResponse:
#     try:
#         # Get Payload Content
#         worker_id = msg.content.get("worker_id", None)
#         _current_user_id = msg.content.get("current_user", None)
#
#         users = node.users
#         if not _current_user_id:
#             _current_user_id = users.first(
#                 verify_key=verify_key.encode(encoder=HexEncoder).decode("utf-8")
#             ).id
#
#         is_admin = users.can_manage_infrastructure(user_id=_current_user_id)
#
#         envs = [
#             int(env.id)
#             for env in node.environments.get_environments(user=_current_user_id)
#         ]
#         created_by_current_user = int(worker_id) in envs
#
#         # Owner / Admin
#         if not is_admin and not created_by_current_user:
#             raise AuthorizationError("You're not allowed to delete this worker!")
#
#         env = node.environments.first(id=worker_id)
#         _config = Config(provider=env.provider, app=Config(name="worker", id=worker_id))
#
#         if env.state == states["success"]:
#             worker_dir = os.path.join(
#                 "/home/ubuntu/.pygrid/apps/aws/workers/", str(worker_id)
#             )
#             success = Provider(worker_dir).destroy()
#             if success:
#                 node.environments.set(
#                     id=worker_id, state=states["destroyed"], destroyed_at=datetime.now()
#                 )
#
#         if env.state == states["destroyed"]:
#             return DeleteWorkerResponse(
#                 address=msg.reply_to,
#                 status_code=200,
#                 content={"message": "Worker was deleted successfully!"},
#             )
#         else:
#             raise Exception("Worker deletion failed")
#
#     except Exception as e:
#         return DeleteWorkerResponse(
#             address=msg.reply_to, status_code=False, content={"error": str(e)}
#         )
#
#
# class DomainInfrastructureService(ImmediateNodeServiceWithReply):
#
#     msg_handler_map = {
#         GetWorkerInstanceTypesMessage: get_worker_instance_types_msg,
#         CreateWorkerMessage: create_worker_msg,
#         GetWorkerMessage: get_worker_msg,
#         GetWorkersMessage: get_workers_msg,
#         DeleteWorkerMessage: del_worker_msg,
#     }
#
#     @staticmethod
#     @service_auth(guests_welcome=True)
#     def process(
#         node: AbstractNode,
#         msg: Union[
#             GetWorkerInstanceTypesMessage,
#             CreateWorkerMessage,
#             GetWorkerMessage,
#             GetWorkersMessage,
#             DeleteWorkerMessage,
#         ],
#         verify_key: VerifyKey,
#     ) -> Union[
#         GetWorkerInstanceTypesResponse,
#         CreateWorkerResponse,
#         GetWorkerResponse,
#         GetWorkersResponse,
#         DeleteWorkerResponse,
#     ]:
#         return DomainInfrastructureService.msg_handler_map[type(msg)](
#             msg=msg, node=node, verify_key=verify_key
#         )
#
#     @staticmethod
#     def message_handler_types() -> List[Type[ImmediateSyftMessageWithReply]]:
#         return [
#             GetWorkerInstanceTypesMessage,
#             CreateWorkerMessage,
#             # CheckWorkerDeploymentMessage,
#             # UpdateWorkerMessage,
#             GetWorkerMessage,
#             GetWorkersMessage,
#             DeleteWorkerMessage,
#         ]
