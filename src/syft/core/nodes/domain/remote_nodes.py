from ..nodes.abstract.remote_nodes import MyRemoteNodes
from ..device.client import DeviceClient

class DomainRemoteNodes(MyRemoteNodes):

    def register_device(self, id_or_name, route):
        self._register_node('device', id_or_name, route)

    def lookup_device(self, id_or_name, route=None):
        devices = self._rotate_on_id(self.nodes['device'])
        return devices[id_or_name]

    def route_message_to_relevant_nodes(self, message):
        pri_route = message.route.pri_route
        device = pri_route.device
        # get a device client and connect.
        connection = route.connect()
        DeviceClient(connection).send(message)
