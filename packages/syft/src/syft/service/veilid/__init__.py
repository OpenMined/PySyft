# stdlib
import os

# relative
from ...util.util import str_to_bool

VEILID_ENABLED: bool = str_to_bool(os.environ.get("VEILID_ENABLED", "False"))
