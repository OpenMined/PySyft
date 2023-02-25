# future
from __future__ import annotations

# stdlib
from functools import partial
import hashlib
import os
from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Type
from typing import Union

# third party
import gevent
import gipc
from gipc.gipc import _GIPCDuplexHandle
from nacl.signing import SigningKey
from result import Err
from result import Result

# relative
from ... import __version__
from ...core.node.common.node_table.syft_object import HIGHEST_SYFT_OBJECT_VERSION
from ...core.node.common.node_table.syft_object import LOWEST_SYFT_OBJECT_VERSION
from ...core.node.common.node_table.syft_object import SyftObject
from ...telemetry import instrument
from ...util import random_name
from ..common.serde.deserialize import _deserialize
from ..common.serde.serializable import serializable
from ..common.serde.serialize import _serialize
from ..common.uid import UID
from .new.action_service import ActionService
from .new.action_store import DictActionStore
from .new.action_store import SQLiteActionStore
from .new.api import SignedSyftAPICall
from .new.api import SyftAPI
from .new.api import SyftAPICall
from .new.context import AuthedServiceContext
from .new.context import NodeServiceContext
from .new.context import UnauthedServiceContext
from .new.context import UserLoginCredentials
from .new.credentials import SyftSigningKey
from .new.data_subject_service import DataSubjectService
from .new.dataset_service import DatasetService
from .new.dict_document_store import DictStoreConfig
from .new.document_store import StoreConfig
from .new.message_service import MessageService
from .new.network_service import NetworkService
from .new.node import NewNode
from .new.node_metadata import NodeMetadata
from .new.queue_stash import QueueItem
from .new.queue_stash import QueueStash
from .new.request_service import RequestService
from .new.response import SyftError
from .new.service import AbstractService
from .new.service import ServiceConfigRegistry
from .new.sqlite_document_store import SQLiteStoreClientConfig
from .new.sqlite_document_store import SQLiteStoreConfig
from .new.test_service import TestService
from .new.user import User
from .new.user import UserCreate
from .new.user_code_service import UserCodeService
from .new.user_service import UserService
from .new.user_stash import UserStash
from .new.worker_settings import WorkerSettings


def gipc_encoder(obj):
    return _serialize(obj, to_bytes=True)


def gipc_decoder(obj_bytes):
    return _deserialize(obj_bytes, from_bytes=True)


NODE_PRIVATE_KEY = "NODE_PRIVATE_KEY"
NODE_UID = "NODE_UID"


def get_private_key_env() -> Optional[str]:
    return get_env(NODE_PRIVATE_KEY)


def get_node_uid_env() -> Optional[str]:
    return get_env(NODE_UID)


def get_env(key: str) -> Optional[str]:
    return os.environ.get(key, None)


signing_key_env = get_private_key_env()
node_uid_env = get_node_uid_env()


@instrument
@serializable(recursive_serde=True)
class Worker(NewNode):
    signing_key: Optional[SyftSigningKey]
    required_signed_calls: bool = True

    def __init__(
        self,
        *,  # Trasterisk
        name: Optional[str] = None,
        id: Optional[UID] = None,
        services: Optional[List[Type[AbstractService]]] = None,
        signing_key: Optional[Union[SyftSigningKey, SigningKey]] = None,
        action_store_config: Optional[StoreConfig] = None,
        document_store_config: Optional[StoreConfig] = None,
        root_email: str = "info@openmined.org",
        root_password: str = "changethis",
        processes: int = 0,
        is_subprocess: bool = False,
    ):
        # 🟡 TODO 22: change our ENV variable format and default init args to make this
        # less horrible or add some convenience functions
        if node_uid_env is not None:
            self.id = UID.from_string(node_uid_env)
        else:
            if id is None:
                id = UID()
            self.id = id

        self.signing_key = None
        if signing_key_env is not None:
            self.signing_key = SyftSigningKey.from_string(signing_key_env)
        else:
            if isinstance(signing_key, SigningKey):
                signing_key = SyftSigningKey(signing_key=signing_key)
            self.signing_key = signing_key

        if self.signing_key is None:
            self.signing_key = SyftSigningKey.generate()

        self.processes = processes
        self.is_subprocess = is_subprocess
        if name is None:
            name = random_name()
        self.name = name
        services = (
            [
                UserService,
                ActionService,
                TestService,
                DatasetService,
                UserCodeService,
                RequestService,
                DataSubjectService,
                NetworkService,
                MessageService,
            ]
            if services is None
            else services
        )
        self.services = services

        self.service_config = ServiceConfigRegistry.get_registered_configs()
        self.init_stores(
            action_store_config=action_store_config,
            document_store_config=document_store_config,
        )
        self._construct_services()
        create_admin_new(  # nosec B106
            name="Jane Doe",
            email="info@openmined.org",
            password="changethis",
            node=self,
        )
        self.client_cache = {}
        self.post_init()

    @staticmethod
    def named(name: str, processes: int = 0) -> Worker:
        name_hash = hashlib.sha256(name.encode("utf8")).digest()
        uid = UID(name_hash[0:16])
        key = SyftSigningKey(SigningKey(name_hash))
        return Worker(name=name, id=uid, signing_key=key, processes=processes)

    @property
    def root_client(self) -> Any:
        # relative
        from .new.client import PythonConnection
        from .new.client import SyftClient

        connection = PythonConnection(node=self)
        return SyftClient(connection=connection, credentials=self.signing_key)

    def __repr__(self) -> str:
        return f"{type(self).__name__}: {self.name} - {self.id} {self.services}"

    def post_init(self) -> None:
        if self.is_subprocess:
            # print(f"> Starting Subprocess {self}")
            pass
        else:
            print(f"> Starting {self}")
        # super().post_init()

    def init_stores(
        self,
        document_store_config: Optional[StoreConfig] = None,
        action_store_config: Optional[StoreConfig] = None,
    ):
        if document_store_config is None:
            if self.processes > 0 and not self.is_subprocess:
                client_config = SQLiteStoreClientConfig()
                document_store_config = SQLiteStoreConfig(client_config=client_config)
            else:
                document_store_config = DictStoreConfig()
        if (
            isinstance(document_store_config, SQLiteStoreConfig)
            and document_store_config.client_config.filename is None
        ):
            document_store_config.client_config.filename
            document_store_config.client_config.filename = f"{self.id}.sqlite"
            print(
                f"SQLite Store Path:\n!open file://{document_store_config.client_config.file_path}\n"
            )
        document_store = document_store_config.store_type
        self.document_store_config = document_store_config

        self.document_store = document_store(store_config=document_store_config)
        if action_store_config is None:
            if self.processes > 0 and not self.is_subprocess:
                client_config = SQLiteStoreClientConfig()
                action_store_config = SQLiteStoreConfig(client_config=client_config)
                if (
                    isinstance(action_store_config, SQLiteStoreConfig)
                    and action_store_config.client_config.filename is None
                ):
                    action_store_config.client_config.filename = f"{self.id}.sqlite"
            else:
                action_store_config = DictStoreConfig()

        if isinstance(action_store_config, SQLiteStoreConfig):
            self.action_store = SQLiteActionStore(
                store_config=action_store_config,
                root_verify_key=self.signing_key.verify_key,
            )
        else:
            self.action_store = DictActionStore(
                root_verify_key=self.signing_key.verify_key
            )

        self.action_store_config = action_store_config
        self.queue_stash = QueueStash(store=self.document_store)

    def _construct_services(self):
        self.service_path_map = {}
        for service_klass in self.services:
            kwargs = {}
            if service_klass == ActionService:
                kwargs["store"] = self.action_store
            if service_klass in [
                UserService,
                DatasetService,
                UserCodeService,
                RequestService,
                DataSubjectService,
                NetworkService,
                MessageService,
            ]:
                kwargs["store"] = self.document_store
            self.service_path_map[service_klass.__name__.lower()] = service_klass(
                **kwargs
            )

    def get_service_method(self, path_or_func: Union[str, Callable]) -> Callable:
        if callable(path_or_func):
            path_or_func = path_or_func.__qualname__
        return self._get_service_method_from_path(path_or_func)

    def get_service(self, path_or_func: Union[str, Callable]) -> Callable:
        if callable(path_or_func):
            path_or_func = path_or_func.__qualname__
        return self._get_service_from_path(path_or_func)

    def _get_service_from_path(self, path: str) -> AbstractService:
        path_list = path.split(".")
        if len(path_list) > 1:
            _ = path_list.pop()
        service_name = path_list.pop()
        return self.service_path_map[service_name.lower()]

    def _get_service_method_from_path(self, path: str) -> Callable:
        path_list = path.split(".")
        method_name = path_list.pop()
        service_obj = self._get_service_from_path(path=path)

        return getattr(service_obj, method_name)

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name=self.name,
            id=self.id,
            verify_key=self.signing_key.verify_key,
            highest_object_version=HIGHEST_SYFT_OBJECT_VERSION,
            lowest_object_version=LOWEST_SYFT_OBJECT_VERSION,
            syft_version=__version__,
        )

    @property
    def icon(self) -> str:
        return "🦾"

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Worker):
            return False

        if self.id != other.id:
            return False

        return True

    def resolve_future(self, uid: UID) -> Union[Optional[QueueItem], SyftError]:
        result = self.queue_stash.pop(uid)
        if result.is_ok():
            return result.ok()
        return result.err()

    def forward_message(
        self, api_call: Union[SyftAPICall, SignedSyftAPICall]
    ) -> Result[Union[QueueItem, SyftObject], Err]:
        node_uid = api_call.message.node_uid
        if NetworkService not in self.services:
            return SyftError(
                message=(
                    "Node has no network service so we can't "
                    f"forward this message to {node_uid}"
                )
            )

        client = None
        if node_uid in self.client_cache:
            client = self.client_cache[node_uid]
        else:
            network_service = self.get_service(NetworkService)
            peer = network_service.stash.get_by_uid(node_uid)

            if peer.is_ok() and peer.ok():
                peer = peer.ok()
                context = NodeServiceContext(node=self)
                client = peer.client_with_context(context=context)
                self.client_cache[node_uid] = client

        if client:
            return client.connection.make_call(api_call)

        return SyftError(message=(f"Node has no route to {node_uid}"))

    def handle_api_call(
        self, api_call: Union[SyftAPICall, SignedSyftAPICall]
    ) -> Result[Union[QueueItem, SyftObject], Err]:
        if self.required_signed_calls and isinstance(api_call, SyftAPICall):
            return SyftError(
                message=f"You sent a {type(api_call)}. This node requires SignedSyftAPICall."  # type: ignore
            )
        else:
            if not api_call.is_valid:
                return SyftError(message="Your message signature is invalid")  # type: ignore

        if api_call.message.node_uid != self.id:
            return self.forward_message(api_call=api_call)

        if api_call.message.path == "queue":
            return self.resolve_future(uid=api_call.message.kwargs["uid"])

        if api_call.message.path == "metadata":
            return self.metadata

        result = None
        if self.is_subprocess or self.processes == 0:
            credentials = api_call.credentials
            api_call = api_call.message

            context = AuthedServiceContext(node=self, credentials=credentials)

            # 🔵 TODO 4: Add @service decorator to autobind services into the SyftAPI
            if api_call.path not in self.service_config:
                return SyftError(message=f"API call not in registered services: {api_call.path}")  # type: ignore

            _private_api_path = ServiceConfigRegistry.private_path_for(api_call.path)
            method = self.get_service_method(_private_api_path)
            try:
                result = method(context, *api_call.args, **api_call.kwargs)
            except Exception as e:
                result = SyftError(message=f"Exception calling {api_call.path}. {e}")
        else:
            worker_settings = WorkerSettings(
                id=self.id,
                name=self.name,
                signing_key=self.signing_key,
                document_store_config=self.document_store_config,
                action_store_config=self.action_store_config,
            )

            task_uid = UID()
            item = QueueItem(id=task_uid, node_uid=self.id)
            # 🟡 TODO 36: Needs distributed lock
            # self.queue_stash.set_placeholder(item)
            # self.queue_stash.partition.commit()
            thread = gevent.spawn(
                queue_task,
                api_call,
                worker_settings,
                task_uid,
                api_call.message.blocking,
            )
            if api_call.message.blocking:
                gevent.joinall([thread])
                result = thread.value
            else:
                result = item
        return result

    def get_api(self) -> SyftAPI:
        return SyftAPI.for_user(node=self)

    def get_method_with_context(
        self, function: Callable, context: NodeServiceContext
    ) -> Callable:
        method = self.get_service_method(function)
        return partial(method, context=context)

    def get_unauthed_context(
        self, login_credentials: UserLoginCredentials
    ) -> NodeServiceContext:
        return UnauthedServiceContext(node=self, login_credentials=login_credentials)


def task_producer(
    pipe: _GIPCDuplexHandle, api_call: SyftAPICall, blocking: bool
) -> Any:
    try:
        result = None
        with pipe:
            pipe.put(api_call)
            gevent.sleep(0)
            if blocking:
                try:
                    result = pipe.get()
                except EOFError:
                    pass
            pipe.close()
        if blocking:
            return result
    except gipc.gipc.GIPCClosed:
        pass
    except Exception as e:
        print("Exception in task_producer", e)


def task_runner(
    pipe: _GIPCDuplexHandle,
    worker_settings: WorkerSettings,
    task_uid: UID,
    blocking: bool,
) -> None:
    worker = Worker(
        id=worker_settings.id,
        name=worker_settings.name,
        signing_key=worker_settings.signing_key,
        document_store_config=worker_settings.document_store_config,
        action_store_config=worker_settings.action_store_config,
        is_subprocess=True,
    )
    try:
        with pipe:
            api_call = pipe.get()
            result = worker.handle_api_call(api_call)
            if blocking:
                pipe.put(result)
            else:
                item = QueueItem(
                    node_uid=worker.id, id=task_uid, result=result, resolved=True
                )
                worker.queue_stash.set_result(item)
                worker.queue_stash.partition.close()
            pipe.close()
    except Exception as e:
        print("Exception in task_runner", e)
        raise e


def queue_task(
    api_call: SyftAPICall,
    worker_settings: WorkerSettings,
    task_uid: UID,
    blocking: bool,
) -> Optional[Any]:
    with gipc.pipe(encoder=gipc_encoder, decoder=gipc_decoder, duplex=True) as (
        cend,
        pend,
    ):
        process = gipc.start_process(
            task_runner, args=(cend, worker_settings, task_uid, blocking)
        )
        producer = gevent.spawn(task_producer, pend, api_call, blocking)
        try:
            process.join()
        except KeyboardInterrupt:
            producer.kill(block=True)
            process.terminate()
        process.join()

    if blocking:
        return producer.value
    return None


def create_admin_new(
    name: str,
    email: str,
    password: str,
    node: NewNode,
) -> Optional[User]:
    try:
        user_stash = UserStash(store=node.document_store)
        row_exists = user_stash.get_by_email(email=email).ok()
        if row_exists:
            return None
        else:
            create_user = UserCreate(
                name=name, email=email, password=password, password_verify=password
            )
            # New User Initialization
            # 🟡 TODO: change later but for now this gives the main user super user automatically
            user = create_user.to(User)
            user.signing_key = node.signing_key
            user.verify_key = user.signing_key.verify_key
            result = user_stash.set(user=user)
            if result.is_ok():
                return result.ok()
            return None
    except Exception as e:
        print("create_admin failed", e)
