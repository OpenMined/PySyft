# Standard Python imports
import json

# Local imports
from ...core.codes import MSG_FIELD


def socket_ping(message: dict) -> str:
    """Ping request to check node's health state."""
    return json.dumps({MSG_FIELD.ALIVE: "True"})
