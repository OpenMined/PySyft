import enum
from .client import Client
from ...io.route import Route
from ...message.syft_message import SyftMessage

class NodeTypes(enum):
    Network = 1
    Domain = 2
    Device = 3
    VM = 4

class Node(object):
    def __init__(self, type, id = None, name = None, route = None, tags=[]):
        self.id = id
        self.name = name
        self.type = type
        self.route = route
        self.tags = tags
        self.is_local = is_local

    def as_dict(self, on_key = 'id'):
        key = getattr(self, on_key)
        if on_key == 'tags':
            key = self._flatten_tags_for_search(key)
        {key: {
            'id': self.id,
            'name': self.name,
            'type': self.type,
            'route': self.route,
            'tags': self.tags}}

    def _flatten_tags_for_search(self):
        return tags.join('-')

class RemoteNodes(object):
    """
    This object keeps a registery of a node's surrounding.
    for example a domain may know a number of public networks,
    a number of other domains and a number of devices.
    this also plays a role in routing messages internally and externally.
    eg. if a domain receives a message from a public network asking to be
    routed to a particular device. the remote nodes would have private routes
    with preconfigured connections that it can route through internally.
    """
    nodes = []

    def __init__(self, my_type):
        self.iam = my_type

    def register_node(self, type: str, id: UID = None, name: str = None,
        route: Route = None, tags: list = None) -> None:
        self.nodes.append(Node(type, id, name, route, tags))

    def forget_node(self, key: str, value: str) -> None:
        #as_dict = self.as_dict(key)
        #del as_dict[value]
        #self.nodes =
        pass

    def as_dict(self, on_key = 'id'):
        all = {}
        for node in nodes:
            all.update(node.as_dict(on_key))
        return all

    def get_node(self, key: str = 'id', value:(str, UID)):
        nodes = self.as_dict(on_key = key)
        return nodes.get(value)

    def route_message_to_relevant_nodes(self, message: SyftMessage) -> None:
        """
        check if the message should be forwarded.
        Network: routes to domains and/or to other networks.
        Domain: routes to devices
        Device: routes to VMs
        VM: doesn't route
        """
        raise NotImplementedError()

    def broadcast(self, route: Route, message: SyftMessage) -> None:
        channel = route.broadcast_channel
        route.connect().send_msg(message)
