from ...decorators import syft_decorator
from ...common.id import UID
from ...common.message import AbstractMessage
from typing import List

class Route(object):
    """
    A route is the highest level interface for how a node can
    reach other nodes. Route provides a name, imagine it like a name server
    so that if user had multiple routes to the same name server
    it can identify it that these routes lead to the same
    destination.

    Route also maintains connection objects which allow for actual
    communication over this route.
    """

    def __init__(self, connection_client_type):
        self.connection_client_type = connection_client_type
        self.connection_client = None

    def connect(self, *args, **kwargs):
        """This method should initialize the connection_client with the correct
        metadata. Args and Kwargs are the connection parameters."""
        return self.connection_client_type(**kwargs)

    def send_msg(self, msg: AbstractMessage):
        raise self.connection_client.send_msg(msg)


class PointToPointRoute(Route):

    def __init__(self, target_node_id: UID, connection_client_type: type):
        super().__init__(connection_client_type=connection_client_type)
        self.target_node_id = target_node_id

class BroadcastRoute(Route):

    #TODO: instead of passing in a list, come up with an abstractoin for
    # "worker group" of arbitrary size and membership which can include
    # "known members" but isn't intended to be exhaustive.
    def __init__(self, target_node_ids: List[UID], connection_client_type: type):
        super().__init__(connection_client_type=connection_client_type)
        self.target_node_ids = target_node_ids

class MQTTRoute(Route):
    def connect(self):
        pass


class HTTPRoute(Route):
    def connect(self):
        pass

