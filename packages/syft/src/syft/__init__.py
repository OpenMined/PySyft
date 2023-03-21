"""
Welcome to the syft package! This package is the primary package for PySyft.
This package has two kinds of attributes: submodules and convenience functions.
Submodules are configured in the standard way, but the convenience
functions exist to allow for a convenient `import syft as sy` to then expose
the most-used functionalities directly on syft. Note that this way of importing
PySyft is the strict convention in this codebase. (Do no simply call
`import syft` and then directly use `syft.<method>`.)
The syft module is split into two distinct groups of functionality which we casually refer to
as syft "core" and syft "python". "core" functionality is functionality which is designed
to be universal across all Syft languages (javascript, kotlin, swift, etc.).
Syft "python" includes all functionality which by its very nature cannot be
truly polyglot. Syft "core" functionality includes the following modules:
* :py:mod:`syft.core.node` - APIs for interacting with remote machines you do not directly control.
* :py:mod:`syft.core.message` - APIs for serializing messages sent between Client and Node classes.
* :py:mod:`syft.core.pointer` - Client side API for referring to objects on a Node
* :py:mod:`syft.core.store` - Server side API for referring to object storage on a node (things pointers point to)
Syft "python" functionality includes the following modules:
* :py:mod:`syft.ast` - code generates external library common syntax tree using an allowlist list of methods
* :py:mod:`syft.typecheck` - automatically checks and enforces Python type hints and the exclusive use of kwargs.
* :py:mod:`syft.lib` - uses the ast library to dynamically create remote execution APIs for supported Python libs.
    IMPORTANT: syft.core should be very careful when importing functionality from outside of syft
    core!!! Since we plan to drop syft core down to a language (such as C++ or Rust)
    this can create future complications with lower level languages calling
    higher level ones.
To begin your education in Syft, continue to the :py:mod:`syft.core.node.vm.vm` module...
"""

__version__ = "0.8.0-beta.3"

# stdlib
from pathlib import Path
import sys
from typing import Any

# relative
from . import filterwarnings  # noqa: F401
from . import gevent_patch  # noqa: F401
from . import jax_settings  # noqa: F401
from . import logger  # noqa: F401
from .core.node.new import NOTHING  # noqa: F401
from .core.node.new.action_object import ActionObject  # noqa: F401
from .core.node.new.client import connect  # noqa: F401
from .core.node.new.client import login  # noqa: F401
from .core.node.new.credentials import SyftSigningKey  # noqa: F401
from .core.node.new.data_subject import DataSubjectCreate as DataSubject  # noqa: F401
from .core.node.new.dataset import CreateAsset as Asset  # noqa: F401
from .core.node.new.dataset import CreateDataset as Dataset  # noqa: F401
from .core.node.new.deserialize import _deserialize as deserialize  # noqa: F401
from .core.node.new.project import ProjectSubmit as Project  # noqa: F401
from .core.node.new.request import SubmitRequest as Request  # noqa: F401
from .core.node.new.response import SyftError  # noqa: F401
from .core.node.new.response import SyftNotReady  # noqa: F401
from .core.node.new.response import SyftSuccess  # noqa: F401
from .core.node.new.roles import Roles as roles  # noqa: F401
from .core.node.new.serialize import _serialize as serialize  # noqa: F401
from .core.node.new.uid import UID  # noqa: F401
from .core.node.new.user_code import ExactMatch  # noqa: F401
from .core.node.new.user_code import SingleExecutionExactOutput  # noqa: F401
from .core.node.new.user_code import UserCodeStatus  # noqa: F401
from .core.node.new.user_code import syft_function  # noqa: F401
from .core.node.new.user_service import UserService  # noqa: F401
from .core.node.worker import Worker  # noqa: F401
from .deploy import Orchestra  # noqa: F401
from .external import OBLV  # noqa: F401
from .external import enable_external_lib  # noqa: F401
from .registry import DomainRegistry  # noqa: F401
from .registry import NetworkRegistry  # noqa: F401
from .search import Search  # noqa: F401
from .search import SearchResults  # noqa: F401
from .telemetry import instrument  # noqa: F401
from .user_settings import UserSettings  # noqa: F401
from .user_settings import settings  # noqa: F401
from .version_compare import make_requires

LATEST_STABLE_SYFT = "0.7"
requires = make_requires(LATEST_STABLE_SYFT, __version__)

sys.path.append(str(Path(__file__)))

logger.start()

# For server-side, to enable by environment variable
if OBLV:
    enable_external_lib("oblv")


def module_property(func: Any) -> None:
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
    return Orchestra()


def search(name: str) -> SearchResults:
    return Search(_domains()).search(name=name)
