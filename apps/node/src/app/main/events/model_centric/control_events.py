# Standard Python imports
import json

# Local imports
from ...core.codes import MSG_FIELD


def socket_ping(message: dict) -> dict:
    """Ping request to check node's health state."""
    return {MSG_FIELD.ALIVE: "True"}
