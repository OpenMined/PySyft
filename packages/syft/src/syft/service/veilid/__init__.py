# stdlib
import os
from typing import Any

# relative
from ...util.util import str_to_bool

VEILID_ENABLED: bool = str_to_bool(os.environ.get("VEILID_ENABLED", "False"))


# Any because circular import
def VeilidServiceProvider(*args: Any, **kwargs: Any) -> Any | None:
    if VEILID_ENABLED:
        # relative
        from .veilid_service import VeilidService

        return VeilidService(*args, **kwargs)
    return None
