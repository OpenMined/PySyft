# -*- coding: utf-8 -*-
"""
    asdfasdfasdfasd
"""

from pkg_resources import get_distribution, DistributionNotFound

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    __version__ = 'unknown'
finally:
    del get_distribution, DistributionNotFound

from syft import ast
from syft import lib
from syft import worker

from syft.worker.virtual.virtual_worker import VirtualWorker
from syft.worker import create_virtual_workers
from syft.worker import create_virtual_workers as cvw