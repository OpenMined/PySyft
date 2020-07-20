import enum
from ...io.route import Route

class RemoteNodeTypes(enum.Enum):
   VM = 1
   Device = 2
   Domain = 3
   Network = 4

class RemoteNode(object):
    def __init__(self, route: Route, type: RemoteNodeType) -> None:
        self.id = id
        self.route = route

class RemoteNodes(object):
    nodes = {}

    def __init__(self):
        for type in RemoteNodeTypes:
            self.nodes.update({type.name: []})

    def register_worker(self, route: Route, type: RemoteNodeType) -> None:
        if type not in self.nodes:
            # log unknown type
            return

        self.nodes[type].append(route)

    def forget_worker(self, route: Route, type: RemoteNodeType) -> None:
        self.nodes[type].pop(route)
