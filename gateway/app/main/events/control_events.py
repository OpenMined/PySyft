import json
from ..codes import RESPONSE_MSG


def socket_ping(message: dict, socket) -> str:
    """ Ping request to check node's health state. """
    return json.dumps({RESPONSE_MSG.ALIVE: "True"})
