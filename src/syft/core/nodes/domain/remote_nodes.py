from ..nodes.abstract.remote_nodes import RemoteNodes
from ..device.client import DeviceClient

class DomainRemoteNodes(RemoteNodes):

    def register_device(self, id_or_name, route):
        self._register_node('device', id_or_name, route)

    def lookup_device(self, id_or_name, route=None):
        devices = self._rotate_on_id(self.nodes['device'])
        return devices[id_or_name]

    def route_message_to_relevant_nodes(self, message):
        """
        this routes the message internally on private connection.
        private network connection is on routes hidden from the outside world.
        """

        pri_address = message.address.pri_route
        device = pri_address.device
        # look up the private connection to this device.
        private_route = self.nodes.get(on_key='name', value=device).route

        # need to make this part work as connection
        #  isn't something that can be done directly on route.
        connection = private_route.connection
        DeviceClient(connection).send_msg(message)
