__version__ = "0.8.0-beta.9"

# stdlib
from pathlib import Path
import sys
from typing import Any
from typing import Callable

# relative
from . import gevent_patch  # noqa: F401
from .client.client import connect  # noqa: F401
from .client.client import login  # noqa: F401
from .client.deploy import Orchestra  # noqa: F401
from .client.registry import DomainRegistry  # noqa: F401
from .client.registry import NetworkRegistry  # noqa: F401
from .client.search import Search  # noqa: F401
from .client.search import SearchResults  # noqa: F401
from .client.user_settings import UserSettings  # noqa: F401
from .client.user_settings import settings  # noqa: F401
from .external import OBLV  # noqa: F401
from .external import enable_external_lib  # noqa: F401
from .node.credentials import SyftSigningKey  # noqa: F401
from .node.domain import Domain  # noqa: F401
from .node.gateway import Gateway  # noqa: F401
from .node.server import serve_node  # noqa: F401
from .node.server import serve_node as bind_worker  # noqa: F401
from .node.worker import Worker  # noqa: F401
from .serde import NOTHING  # noqa: F401
from .serde.deserialize import _deserialize as deserialize  # noqa: F401
from .serde.serializable import serializable  # noqa: F401
from .serde.serialize import _serialize as serialize  # noqa: F401
from .service.action.action_object import ActionObject  # noqa: F401
from .service.code.user_code import UserCodeStatus  # noqa: F401
from .service.code.user_code import syft_function  # noqa: F401
from .service.data_subject import DataSubjectCreate as DataSubject  # noqa: F401
from .service.dataset.dataset import CreateAsset as Asset  # noqa: F401
from .service.dataset.dataset import CreateDataset as Dataset  # noqa: F401
from .service.message.messages import MessageStatus  # noqa: F401
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
from .types.uid import UID  # noqa: F401
from .util import filterwarnings  # noqa: F401
from .util import jax_settings  # noqa: F401
from .util import logger  # noqa: F401
from .util.telemetry import instrument  # noqa: F401
from .util.util import autocache  # noqa: F401
from .util.version_compare import make_requires

LATEST_STABLE_SYFT = "0.7"
requires = make_requires(LATEST_STABLE_SYFT, __version__)

sys.path.append(str(Path(__file__)))

logger.start()

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
