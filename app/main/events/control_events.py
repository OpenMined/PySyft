import json
from ..codes import MSG_FIELD


def socket_ping(message: dict, socket) -> str:
    """ Ping request to check node's health state. """
    return json.dumps({MSG_FIELD.ALIVE: "True"})
