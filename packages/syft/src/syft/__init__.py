__version__ = "0.8.5-post.2"

# stdlib
from collections.abc import Callable
from getpass import getpass
import pathlib
from pathlib import Path
import random
import sys
from typing import Any

# relative
from . import gevent_patch  # noqa: F401
from .abstract_node import NodeSideType  # noqa: F401
from .abstract_node import NodeType  # noqa: F401
from .client.client import connect  # noqa: F401
from .client.client import login  # noqa: F401
from .client.client import login_as_guest  # noqa: F401
from .client.client import register  # noqa: F401
from .client.deploy import Orchestra  # noqa: F401
from .client.domain_client import DomainClient  # noqa: F401
from .client.gateway_client import GatewayClient  # noqa: F401
from .client.registry import DomainRegistry  # noqa: F401
from .client.registry import EnclaveRegistry  # noqa: F401
from .client.registry import NetworkRegistry  # noqa: F401
from .client.search import Search  # noqa: F401
from .client.search import SearchResults  # noqa: F401
from .client.user_settings import UserSettings  # noqa: F401
from .client.user_settings import settings  # noqa: F401
from .custom_worker.config import DockerWorkerConfig  # noqa: F401
from .external import OBLV_ENABLED  # noqa: F401
from .external import enable_external_lib  # noqa: F401
from .node.credentials import SyftSigningKey  # noqa: F401
from .node.domain import Domain  # noqa: F401
from .node.enclave import Enclave  # noqa: F401
from .node.gateway import Gateway  # noqa: F401
from .node.server import serve_node  # noqa: F401
from .node.server import serve_node as bind_worker  # noqa: F401
from .node.worker import Worker  # noqa: F401
from .protocol.data_protocol import bump_protocol_version  # noqa: F401
from .protocol.data_protocol import check_or_stage_protocol  # noqa: F401
from .protocol.data_protocol import get_data_protocol  # noqa: F401
from .protocol.data_protocol import stage_protocol_changes  # noqa: F401
from .serde import NOTHING  # noqa: F401
from .serde.deserialize import _deserialize as deserialize  # noqa: F401
from .serde.serializable import serializable  # noqa: F401
from .serde.serialize import _serialize as serialize  # noqa: F401
from .service.action.action_data_empty import ActionDataEmpty  # noqa: F401
from .service.action.action_object import ActionObject  # noqa: F401
from .service.action.plan import Plan  # noqa: F401
from .service.action.plan import planify  # noqa: F401
from .service.code.user_code import UserCodeStatus  # noqa: F401; noqa: F401
from .service.code.user_code import syft_function  # noqa: F401; noqa: F401
from .service.code.user_code import syft_function_single_use  # noqa: F401; noqa: F401
from .service.data_subject import DataSubjectCreate as DataSubject  # noqa: F401
from .service.dataset.dataset import Contributor  # noqa: F401
from .service.dataset.dataset import CreateAsset as Asset  # noqa: F401
from .service.dataset.dataset import CreateDataset as Dataset  # noqa: F401
from .service.notification.notifications import NotificationStatus  # noqa: F401
from .service.policy.policy import CustomInputPolicy  # noqa: F401
from .service.policy.policy import CustomOutputPolicy  # noqa: F401
from .service.policy.policy import ExactMatch  # noqa: F401
from .service.policy.policy import SingleExecutionExactOutput  # noqa: F401
from .service.policy.policy import UserInputPolicy  # noqa: F401
from .service.policy.policy import UserOutputPolicy  # noqa: F401
from .service.project.project import ProjectSubmit as Project  # noqa: F401
from .service.request.request import SubmitRequest as Request  # noqa: F401
from .service.response import SyftError  # noqa: F401
from .service.response import SyftNotReady  # noqa: F401
from .service.response import SyftSuccess  # noqa: F401
from .service.user.roles import Roles as roles  # noqa: F401
from .service.user.user_service import UserService  # noqa: F401
from .stable_version import LATEST_STABLE_SYFT
from .types.twin_object import TwinObject  # noqa: F401
from .types.uid import UID  # noqa: F401
from .util import filterwarnings  # noqa: F401
from .util import jax_settings  # noqa: F401
from .util import logger  # noqa: F401
from .util import options  # noqa: F401
from .util.autoreload import disable_autoreload  # noqa: F401
from .util.autoreload import enable_autoreload  # noqa: F401
from .util.telemetry import instrument  # noqa: F401
from .util.util import autocache  # noqa: F401
from .util.util import get_root_data_path  # noqa: F401
from .util.version_compare import make_requires

requires = make_requires(LATEST_STABLE_SYFT, __version__)


# SYFT_PATH = path = os.path.abspath(a_module.__file__)
SYFT_PATH = pathlib.Path(__file__).parent.resolve()

sys.path.append(str(Path(__file__)))

logger.start()

try:
    # third party
    from IPython import get_ipython

    get_ipython()  # noqa: F821
    # TODO: add back later or auto detect
    # display(
    #     Markdown(
    #         "\nWarning: syft is imported in light mode by default. \
    #     \nTo switch to dark mode, please run `sy.options.color_theme = 'dark'`"
    #     )
    # )
except:  # noqa: E722
    pass  # nosec

# For server-side, to enable by environment variable
if OBLV_ENABLED:
    enable_external_lib("oblv")


def module_property(func: Any) -> Callable:
    """Decorator to turn module functions into properties.
    Function names must be prefixed with an underscore."""
    module = sys.modules[func.__module__]

    def base_getattr(name: str) -> None:
        raise AttributeError(f"module {module.__name__!r} has no attribute {name!r}")

    old_getattr = getattr(module, "__getattr__", base_getattr)

    def new_getattr(name: str) -> Any:
        if f"_{name}" == func.__name__:
            return func()
        else:
            return old_getattr(name)

    module.__getattr__ = new_getattr  # type: ignore
    return func


@module_property
def _gateways() -> NetworkRegistry:
    return NetworkRegistry()


@module_property
def _enclaves() -> EnclaveRegistry:
    return EnclaveRegistry()


@module_property
def _domains() -> DomainRegistry:
    return DomainRegistry()


@module_property
def _settings() -> UserSettings:
    return settings


@module_property
def _orchestra() -> Orchestra:
    return Orchestra


def search(name: str) -> SearchResults:
    return Search(_domains()).search(name=name)


def launch(
    # node information and deployment
    name: str | None = None,
    node_type: str | NodeType | None = None,
    deploy_to: str | None = None,
    node_side_type: str | None = None,
    # worker related inputs
    port: int | str | None = "auto",
    processes: int = 1,  # temporary work around for jax in subprocess
    local_db: bool = False,
    dev_mode: bool = False,
    cmd: bool = False,
    reset: bool = False,
    tail: bool = False,
    host: str | None = "0.0.0.0",  # nosec
    tag: str | None = "latest",
    verbose: bool = False,
    render: bool = False,
    enable_warnings: bool = False,
    n_consumers: int = 0,
    thread_workers: bool = False,
    create_producer: bool = False,
    queue_port: int | None = None,
    in_memory_workers: bool = True,
    skip_signup: bool = False,
    default_admin: bool = False,
) -> DomainClient:
    node = Orchestra.launch(
        name=name,
        node_type=node_type,
        deploy_to=deploy_to,
        node_side_type=node_side_type,
        port=port,
        processes=processes,
        local_db=local_db,
        dev_mode=dev_mode,
        cmd=cmd,
        reset=reset,
        tail=tail,
        host=host,
        tag=tag,
        verbose=verbose,
        render=render,
        enable_warnings=enable_warnings,
        n_consumers=n_consumers,
        thread_workers=thread_workers,
        create_producer=create_producer,
        queue_port=queue_port,
        in_memory_workers=in_memory_workers,
    )

    client = node.login(
        email="info@openmined.org",
        password="changethis",  # nosec
        verbose=False,
        suppress_warnings=not default_admin,
    )

    # so that the user doesn't need to keep up with a node_handlne and client
    client.land = node.land

    if not skip_signup:
        if not default_admin:
            print("\nConfiguring admin account...")
            name = input("\tAdmin Name:")
            email = input("\tAdmin Email:")
            password = getpass("\tAdmin Password:")

            client.me.set_name(name)
            client.me.set_email(email)
            client.me.set_password(password)

        # set default email notifier (note: each email account can only send 300 emails a day,
        # and it might go to people's junk folders.)
        emails_and_passwords = list()  # noqa: C408
        emails_and_passwords.append(
            ("syftforwarder@outlook.com", "Wh8fHys!Nrw7VyXLMj7r")
        )
        emails_and_passwords.append(
            ("andrewtrask1@outlook.com", "notasecurepassword123")
        )

        sender, sender_pwd = random.choice(emails_and_passwords)  # nosec
        client.settings.enable_notifications(
            email_username=sender,
            email_password=sender_pwd,
            email_sender=sender,
            email_server="smtp-mail.outlook.com",
            email_port="587",
        )
        if not default_admin:
            print("Launched and configured!\n")
        return client

    return client
