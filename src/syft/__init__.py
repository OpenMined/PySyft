# -*- coding: utf-8 -*-
"""
Welcome to the syft package! This package is the primary package for PySyft.
This package has two kinds of attributes: submodules, and convenience functions.
Submodules are configured in the standard way, but the convenience
functions exist to allow for a convenient `import syft as sy` to then expose
the most-used functionalities directly on syft. Note that this way of importing
PySyft is the strict convention in this codebase. (Do no simply call
`import syft` and then directly use `syft.<method>`.)

The syft module is split into two
distinct groups of functionality which we casually refer to as syft "core"
and syft "python". "core" functionality is functionality which is designed
to be universal across all Syft languages (javascript, kotlin, swift, etc.).
Syft "python" includes all functionality which by its very nature cannot be
truly polyglot. Syft "core" functionality includes the following modules:

* :py:mod:`syft.core.worker` - APIs for interacting with remote machines you do not directly control.
* :py:mod:`syft.core.message` - APIs for serializing messages sent between Client and Worker classes.
* :py:mod:`syft.core.pointer` - Client side API for referring to objects on a Worker
* :py:mod:`syft.core.store` - Server side API for referring to object storage on a worker (things pointers point to)

Syft "python" functionality includes the following modules:

* :py:mod:`syft.ast` - code generates external library abstract syntax tree using a white\
list of methods
* :py:mod:`syft.typecheck` - automatically checks Python type hints
* :py:mod:`syft.lib` - uses the ast library to dynamically create remote execution APIs for supported Python libs.

IMPORTANT: syft.core should NEVER import functionality from outside of syft
core!!! Since we plan to drop syft core down to a language (such as C++ or Rust)
this would create future complications with lower level languages calling
higher level ones.
"""

from pkg_resources import get_distribution, DistributionNotFound

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    __version__ = "unknown"
finally:
    del get_distribution, DistributionNotFound

# from . import message
# from . import pointer
# from . import ast
# from . import worker
# from . import lib

# from .worker.virtual.virtual_worker import VirtualWorker
# from .worker import create_virtual_workers
# from .worker import create_virtual_workers as cvw
