__version__ = "0.8.4-beta.29"

# stdlib
import pathlib
from pathlib import Path
import sys
from typing import Any
from typing import Callable

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
from .external import OBLV  # noqa: F401
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

LATEST_STABLE_SYFT = "0.8.3"
requires = make_requires(LATEST_STABLE_SYFT, __version__)


# SYFT_PATH = path = os.path.abspath(a_module.__file__)
SYFT_PATH = pathlib.Path(__file__).parent.resolve()

sys.path.append(str(Path(__file__)))

logger.start()

try:
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
if OBLV:
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
