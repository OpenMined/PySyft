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

# stdlib
from pathlib import Path
import sys
from typing import Any
from typing import Dict

# third party
from pkg_resources import DistributionNotFound  # noqa: F401
from pkg_resources import get_distribution  # noqa: F401

# syft absolute
# ASTRACT OBJECT IMPORTS
from syft.core import common  # noqa: F401
from syft.core.common import event_loop  # noqa: F401

# Convenience Methods
from syft.core.common.serde.deserialize import _deserialize as deserialize  # noqa: F401
from syft.core.common.serde.serialize import _serialize as serialize  # noqa: F401
from syft.core.node.common.service.repr_service import ReprMessage  # noqa: F401
from syft.core.node.device.device import Device  # noqa: F401
from syft.core.node.device.device import DeviceClient  # noqa: F401
from syft.core.node.domain.domain import Domain  # noqa: F401
from syft.core.node.domain.domain import DomainClient  # noqa: F401
from syft.core.node.network.network import Network  # noqa: F401
from syft.core.node.network.network import NetworkClient  # noqa: F401

# Convenience Constructors
from syft.core.node.vm.vm import VirtualMachine  # noqa: F401
from syft.core.node.vm.vm import VirtualMachineClient  # noqa: F401
from syft.core.plan.plan import Plan  # noqa: F401
from syft.core.plan.plan_builder import make_plan  # noqa: F401
from syft.experimental_flags import flags  # noqa: F401
from syft.grid.client.client import login  # noqa: F401

# Convenience Functions
from syft.grid.duet import bcolors  # noqa: F401
from syft.grid.duet import duet  # noqa: F401
from syft.grid.duet import join_duet  # noqa: F401
from syft.grid.duet import launch_duet  # noqa: F401

# Convenience Objects
from syft.lib import lib_ast  # noqa: F401
from syft.lib import load  # noqa: F401
from syft.lib import load_lib  # noqa: F401
from syft.lib.torch.module import Module  # noqa: F401
from syft.lib.torch.module import SyModule  # noqa: F401
from syft.lib.torch.module import SySequential  # noqa: F401

# syft relative
# Package Imports
from . import lib  # noqa: F401
from . import logger  # noqa: F401

if sys.version_info[:2] >= (3, 8):
    # TODO: Import directly (no need for conditional) when `python_requires = >= 3.8`
    # stdlib
    from importlib.metadata import PackageNotFoundError  # pragma: no cover
    from importlib.metadata import version
else:
    # third party
    from importlib_metadata import PackageNotFoundError  # pragma: no cover
    from importlib_metadata import version

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

sys.path.append(str(Path(__file__)))

logger.add(sink=sys.stderr, level="CRITICAL")

# TODO: remove this requirement in pytorch lightning
client_cache: Dict[str, Any] = {}
