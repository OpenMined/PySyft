# -*- coding: utf-8 -*-
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

* :py:mod:`syft.core.node` - APIs for interacting with remote machines you do not directly
control.
* :py:mod:`syft.core.message` - APIs for serializing messages sent between Client and Node
classes.
* :py:mod:`syft.core.pointer` - Client side API for referring to objects on a Node
* :py:mod:`syft.core.store` - Server side API for referring to object storage on a node
(things pointers point to)

Syft "python" functionality includes the following modules:

* :py:mod:`syft.ast` - code generates external library abstract syntax tree using a white\
list of methods
* :py:mod:`syft.typecheck` - automatically checks and enforces Python type hints and the exclusive
use of kwargs.
* :py:mod:`syft.lib` - uses the ast library to dynamically create remote execution APIs for
supported Python libs.

    IMPORTANT: syft.core should be very careful when importing functionality from outside of syft
    core!!! Since we plan to drop syft core down to a language (such as C++ or Rust)
    this can create future complications with lower level languages calling
    higher level ones.

To begin your education in Syft, continue to the :py:mod:`syft.core.nodes.vm.vm` module...
"""


# ASTRACT OBJECT IMPORTS
from . import common  # noqa: F401


# CONVENIENCE FUNCTIONS
from .decorators import type_hints  # noqa: F401

from pkg_resources import get_distribution, DistributionNotFound  # noqa: F401


# PACKAGE IMPORTS
from . import lib  # noqa: F401


# VERSIONING
try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    __version__ = "unknown"
finally:
    del get_distribution, DistributionNotFound

from syft.core.nodes.vm.vm import VirtualMachine
from syft.core.nodes.device.device import Device
from syft.core.nodes.domain.domain import Domain
from syft.core.nodes.network.network import Network

from syft.core.nodes.abstract.service.repr_service import ReprMessage

# def get_client(host="127.0.0.1", port="5000"):
#     res = requests.get(f"http://{host}:{port}/")
#     client = pickle.loads(bytes.fromhex(res.text))
#     return client
