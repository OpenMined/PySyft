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

__version__ = "0.7.0-beta.52"

# stdlib
from pathlib import Path
import sys
from typing import Any

# third party
from pkg_resources import DistributionNotFound  # noqa: F401
from pkg_resources import get_distribution  # noqa: F401

# relative
# Package Imports
from . import filterwarnings  # noqa: F401
from . import jax_settings  # noqa: F401
from . import lib  # noqa: F401
from . import logger  # noqa: F401

# ASTRACT OBJECT IMPORTS
from .core import common  # noqa: F401
from .core.adp.data_subject_list import DataSubjectArray  # noqa: F401
from .core.adp.data_subject_list import DataSubjectList  # noqa: F401

# Convenience Methods
from .core.common.serde.deserialize import _deserialize as deserialize  # noqa: F401
from .core.common.serde.serialize import _serialize as serialize  # noqa: F401

# TFF
from .core.node.common.node_service import tff  # noqa: F401
from .core.node.common.node_service.testing_services.repr_service import (  # noqa: F401
    ReprMessage,
)
from .core.node.device import Device  # noqa: F401
from .core.node.device_client import DeviceClient  # noqa: F401
from .core.node.domain import Domain  # noqa: F401
from .core.node.domain import DomainClient  # noqa: F401
from .core.node.network import Network  # noqa: F401
from .core.node.network_client import NetworkClient  # noqa: F401

# Convenience Constructors
from .core.node.vm import VirtualMachine  # noqa: F401
from .core.node.vm_client import VirtualMachineClient  # noqa: F401
from .core.tensor import autodp  # noqa: F401
from .core.tensor import nn  # noqa: F401
from .core.tensor.autodp.gamma_tensor import GammaTensor  # noqa: F401
from .core.tensor.autodp.phi_tensor import PhiTensor  # noqa: F401
from .core.tensor.lazy_repeat_array import lazyrepeatarray  # noqa: F401
from .core.tensor.tensor import Tensor  # noqa: F401
from .experimental_flags import flags  # noqa: F401
from .grid.client.client import connect  # noqa: F401
from .grid.client.client import login  # noqa: F401
from .grid.client.client import register  # noqa: F401

# Convenience Objects
from .lib import lib_ast  # noqa: F401
from .lib import load  # noqa: F401
from .lib import load_lib  # noqa: F401
from .registry import NetworkRegistry  # noqa: F401

sys.path.append(str(Path(__file__)))

logger.start()


def module_property(func: Any) -> None:
    """Decorator to turn module functions into properties.
    Function names must be prefixed with an underscore."""
    module = sys.modules[func.__module__]

    def base_getattr(name: str) -> None:
        raise AttributeError(f"module '{module.__name__}' has no attribute '{name}'")

    old_getattr = getattr(module, "__getattr__", base_getattr)

    def new_getattr(name: str) -> Any:
        if f"_{name}" == func.__name__:
            return func()
        else:
            return old_getattr(name)

    module.__getattr__ = new_getattr  # type: ignore
    return func


@module_property
def _networks() -> NetworkRegistry:
    return NetworkRegistry()
