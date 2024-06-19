__version__ = "0.8.7-beta.10"

# stdlib
from collections.abc import Callable
import pathlib
from pathlib import Path
import sys
from types import MethodType
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
from .service.notification.notifications import NotificationStatus
from .service.policy.policy import CustomInputPolicy
from .service.policy.policy import CustomOutputPolicy
from .service.policy.policy import ExactMatch
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
from .types.syft_object import SyftObject
from .types.twin_object import TwinObject
from .types.uid import UID
from .util import filterwarnings
from .util import options
from .util.autoreload import disable_autoreload
from .util.autoreload import enable_autoreload
from .util.telemetry import instrument
from .util.util import autocache
from .util.util import get_root_data_path
from .util.version_compare import make_requires

requires = make_requires(LATEST_STABLE_SYFT, __version__)


# SYFT_PATH = path = os.path.abspath(a_module.__file__)
SYFT_PATH = pathlib.Path(__file__).parent.resolve()

sys.path.append(str(Path(__file__)))


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


def _patch_ipython_autocompletion() -> None:
    try:
        # third party
        from IPython.core.guarded_eval import EVALUATION_POLICIES
    except ImportError:
        return

    ipython = get_ipython()
    if ipython is None:
        return

    try:
        # this allows property getters to be used in nested autocomplete
        ipython.Completer.evaluation = "limited"
        ipython.Completer.use_jedi = False
        policy = EVALUATION_POLICIES["limited"]

        policy.allowed_getattr_external.update(
            [
                ("syft.client.api", "APIModule"),
                ("syft.client.api", "SyftAPI"),
            ]
        )
        original_can_get_attr = policy.can_get_attr

        def patched_can_get_attr(value: Any, attr: str) -> bool:
            attr_name = "__syft_allow_autocomplete__"
            # first check if exist to prevent side effects
            if hasattr(value, attr_name) and attr in getattr(value, attr_name, []):
                if attr in dir(value):
                    return True
                else:
                    return False
            else:
                return original_can_get_attr(value, attr)

        policy.can_get_attr = patched_can_get_attr
    except Exception:
        print("Failed to patch ipython autocompletion for syft property getters")

    try:
        # this constraints the completions for autocomplete.
        # if __syft_dir__ is defined we only autocomplete those properties
        # stdlib
        import re

        original_attr_matches = ipython.Completer.attr_matches

        def patched_attr_matches(self, text: str) -> list[str]:  # type: ignore
            res = original_attr_matches(text)
            m2 = re.match(r"(.+)\.(\w*)$", self.line_buffer)
            if not m2:
                return res
            expr, _ = m2.group(1, 2)
            obj = self._evaluate_expr(expr)
            if isinstance(obj, SyftObject) and hasattr(obj, "__syft_dir__"):
                # here we filter all autocomplete results to only contain those
                # defined in __syft_dir__, however the original autocomplete prefixes
                # have the full path, while __syft_dir__ only defines the attr
                attrs = set(obj.__syft_dir__())
                new_res = []
                for r in res:
                    splitted = r.split(".")
                    if len(splitted) > 1:
                        attr_name = splitted[-1]
                        if attr_name in attrs:
                            new_res.append(r)
                return new_res
            else:
                return res

        ipython.Completer.attr_matches = MethodType(
            patched_attr_matches, ipython.Completer
        )
    except Exception:
        print("Failed to patch syft autocompletion for __syft_dir__")


_patch_ipython_autocompletion()


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
