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

* :py:mod:`syft.worker` - APIs for interacting with remote machines you do not directly control.
* :py:mod:`syft.message` - APIs for serializing messages sent between Client and Worker classes.
* :py:mod:`syft.pointer` - Client side API for referring to objects on a Worker

Syft "python" functionality includes the following modules:

* :py:mod:`syft.ast` - code generates external library abstract syntax tree using a white\
list of methods
* :py:mod:`syft.typecheck` - automatically checks Python type hints


This is a skeleton file that can serve as a starting point for a Python
console script. To run this script uncomment the following lines in the
[options.entry_points] section in setup.cfg:

    console_scripts =
         fibonacci = syft.skeleton:run

Then run `python setup.py install` which will install the command `fibonacci`
inside your current environment.
Besides console scripts, the header (i.e. until _logger...) of this file can
also be used as template for Python modules.

Note: This skeleton file can be safely removed if not needed!
"""
from . import worker, typecheck, message


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
# from . import lib

# from .worker.virtual.virtual_worker import VirtualWorker
# from .worker import create_virtual_workers
# from .worker import create_virtual_workers as cvw
