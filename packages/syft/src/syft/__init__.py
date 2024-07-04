__version__ = "0.8.7-beta.13"

# stdlib
from collections.abc import Callable
import pathlib
from pathlib import Path
import sys
from typing import Any

# relative
from .abstract_node import NodeSideType
from .abstract_node import NodeType
from .client.client import connect
from .client.client import login
from .client.client import login_as_guest
from .client.client import register
from .client.domain_client import DomainClient
from .client.gateway_client import GatewayClient
from .client.registry import DomainRegistry
from .client.registry import EnclaveRegistry
from .client.registry import NetworkRegistry
from .client.search import Search
from .client.search import SearchResults
from .client.syncing import compare_clients
from .client.syncing import compare_states
from .client.syncing import sync
from .client.user_settings import UserSettings
from .client.user_settings import settings
from .custom_worker.config import DockerWorkerConfig
from .custom_worker.config import PrebuiltWorkerConfig
from .node.credentials import SyftSigningKey
from .node.domain import Domain
from .node.enclave import Enclave
from .node.gateway import Gateway
from .node.server import serve_node
from .node.server import serve_node as bind_worker
from .node.worker import Worker
from .orchestra import Orchestra as orchestra
from .protocol.data_protocol import bump_protocol_version
from .protocol.data_protocol import check_or_stage_protocol
from .protocol.data_protocol import get_data_protocol
from .protocol.data_protocol import stage_protocol_changes
from .serde import NOTHING
from .serde.deserialize import _deserialize as deserialize
from .serde.serializable import serializable
from .serde.serialize import _serialize as serialize
from .service.action.action_data_empty import ActionDataEmpty
from .service.action.action_object import ActionObject
from .service.action.plan import Plan
from .service.action.plan import planify
from .service.api.api import api_endpoint
from .service.api.api import api_endpoint_method
from .service.api.api import create_new_api_endpoint as TwinAPIEndpoint
from .service.code.user_code import UserCodeStatus
from .service.code.user_code import syft_function
from .service.code.user_code import syft_function_single_use
from .service.data_subject import DataSubjectCreate as DataSubject
from .service.dataset.dataset import Contributor
from .service.dataset.dataset import CreateAsset as Asset
from .service.dataset.dataset import CreateDataset as Dataset
from .service.model.model import CreateModel as Model
from .service.model.model import CreateModelAsset as ModelAsset
from .service.notification.notifications import NotificationStatus
from .service.policy.policy import CreatePolicyRuleConstant as Constant
from .service.policy.policy import CustomInputPolicy
from .service.policy.policy import CustomOutputPolicy
from .service.policy.policy import ExactMatch
from .service.policy.policy import MixedInputPolicy
from .service.policy.policy import SingleExecutionExactOutput
from .service.policy.policy import UserInputPolicy
from .service.policy.policy import UserOutputPolicy
from .service.project.project import ProjectSubmit as Project
from .service.request.request import SubmitRequest as Request
from .service.response import SyftError
from .service.response import SyftNotReady
from .service.response import SyftSuccess
from .service.user.roles import Roles as roles
from .service.user.user_service import UserService
from .stable_version import LATEST_STABLE_SYFT
from .types.file import SyftFolder
from .types.twin_object import TwinObject
from .types.uid import UID
from .util import filterwarnings
from .util import options
from .util.autoreload import disable_autoreload
from .util.autoreload import enable_autoreload
from .util.patch_ipython import patch_ipython
from .util.telemetry import instrument
from .util.util import autocache
from .util.util import get_root_data_path
from .util.version_compare import make_requires

requires = make_requires(LATEST_STABLE_SYFT, __version__)


# SYFT_PATH = path = os.path.abspath(a_module.__file__)
SYFT_PATH = pathlib.Path(__file__).parent.resolve()

sys.path.append(str(Path(__file__)))


patch_ipython()


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
def hello_baby() -> None:
    print("Hello baby!")
    print("Welcome to the world. \u2764\ufe0f")


def search(name: str) -> SearchResults:
    return Search(_domains()).search(name=name)
