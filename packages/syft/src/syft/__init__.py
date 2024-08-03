__version__ = "0.9.1-beta.1"

# stdlib
import pathlib
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

# relative
from .abstract_server import ServerSideType, ServerType
from .client.client import connect, login, login_as_guest, register
from .client.datasite_client import DatasiteClient
from .client.gateway_client import GatewayClient
from .client.registry import DatasiteRegistry, EnclaveRegistry, NetworkRegistry
from .client.search import Search, SearchResults
from .client.syncing import compare_clients, compare_states, sync
from .client.user_settings import UserSettings, settings
from .custom_worker.config import DockerWorkerConfig, PrebuiltWorkerConfig
from .orchestra import Orchestra as orchestra
from .protocol.data_protocol import (
    bump_protocol_version,
    check_or_stage_protocol,
    get_data_protocol,
    stage_protocol_changes,
)
from .serde import NOTHING
from .serde.deserialize import _deserialize as deserialize
from .serde.serializable import serializable
from .serde.serialize import _serialize as serialize
from .server.credentials import SyftSigningKey
from .server.datasite import Datasite
from .server.enclave import Enclave
from .server.gateway import Gateway
from .server.uvicorn import serve_server
from .server.uvicorn import serve_server as bind_worker
from .server.worker import Worker
from .service.action.action_data_empty import ActionDataEmpty
from .service.action.action_object import ActionObject
from .service.action.plan import Plan, planify
from .service.api.api import api_endpoint, api_endpoint_method
from .service.api.api import create_new_api_endpoint as TwinAPIEndpoint
from .service.code.user_code import (
    UserCodeStatus,
    syft_function,
    syft_function_single_use,
)
from .service.data_subject import DataSubjectCreate as DataSubject
from .service.dataset.dataset import Contributor
from .service.dataset.dataset import CreateAsset as Asset
from .service.dataset.dataset import CreateDataset as Dataset
from .service.notification.notifications import NotificationStatus
from .service.policy.policy import CreatePolicyRuleConstant as Constant
from .service.policy.policy import (
    CustomInputPolicy,
    CustomOutputPolicy,
    ExactMatch,
    MixedInputPolicy,
    SingleExecutionExactOutput,
    UserInputPolicy,
    UserOutputPolicy,
)
from .service.project.project import ProjectSubmit as Project
from .service.request.request import SubmitRequest as Request
from .service.response import SyftError, SyftNotReady, SyftSuccess
from .service.user.roles import Roles as roles
from .service.user.user_service import UserService
from .stable_version import LATEST_STABLE_SYFT
from .types.twin_object import TwinObject
from .types.uid import UID
from .util import filterwarnings, options
from .util.autoreload import disable_autoreload, enable_autoreload
from .util.commit import __commit__
from .util.patch_ipython import patch_ipython
from .util.telemetry import instrument
from .util.util import autocache, get_nb_secrets, get_root_data_path
from .util.version_compare import make_requires

requires = make_requires(LATEST_STABLE_SYFT, __version__)


# SYFT_PATH = path = os.path.abspath(a_module.__file__)
SYFT_PATH = pathlib.Path(__file__).parent.resolve()

sys.path.append(str(Path(__file__)))


patch_ipython()


def module_property(func: Any) -> Callable:
    """Decorator to turn module functions into properties.
    Function names must be prefixed with an underscore.
    """
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
def _datasites() -> DatasiteRegistry:
    return DatasiteRegistry()


@module_property
def _settings() -> UserSettings:
    return settings


@module_property
def hello_baby() -> None:
    print("Hello baby!")
    print("Welcome to the world. \u2764\ufe0f")


def search(name: str) -> SearchResults:
    return Search(_datasites()).search(name=name)
