from ...decorators import syft_decorator
from ...common.id import UID
from typing import final


class Route(object):
    """
    A route is how a node can be reached by other nodes.
    Route provides a name, imagine it like a name server
        so that if user had multiple routes to the same name server
        it can identify it that these routes lead to the same
        destination.
    and Route provides connection details which could be any protocol.
    potentially this could also serve authentication details.
    """
    def __init__(self, node_unique_name: UID):
        self.name = node_unique_name
        self.connection_details = {}

    def configure_connection(self, protocol: str, host: str, port: int):
        """
        the route should have connection details embedded in it.
        so that nodes operators can utilize it to route messages.
        """
        self.connection_details.update({
            'host': host,
            'protocol': protocol,
            'port': port
        })

    def register_broadcast_channel(self, channel_name: str):
        """
        In the case configured protocol is pub/sub or event driven.
        Args:
            channel_name: the name of the channel to broadcast on.
        """
        self.connection_details.update({'broadcast_channel': channel_name})


class MQTTRoute(Route):
    def connect(self):
        pass

class HTTPRoute(Route):
    def connect(self):
        pass
