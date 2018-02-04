"""Python IPFS API client library"""

from __future__ import absolute_import

from .version import __version__

###########################
# Import stable API parts #
###########################
from . import exceptions

from .client import DEFAULT_HOST, DEFAULT_PORT, DEFAULT_BASE
from .client import VERSION_MINIMUM, VERSION_MAXIMUM
from .client import Client, assert_version, connect
