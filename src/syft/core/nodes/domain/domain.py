from ..abstract.node import Node
from ...message.syft_message import SyftMessage
from ....decorators.syft_decorator import syft_decorator
from .client import DomainClient
from ...io.virtual import create_virtual_connection
from ..device.device import Device
from ..device.client import DeviceClient


class Domain(Node):

    child_type = Device
    child_type_client_type = DeviceClient

    def __init__(self, name):
        super().__init__(name=name)
        # available_device_types = set()
        # TODO: add available compute types

        # default_device = None
        # TODO: add default compute type

        self._register_services()

    def message_is_for_me(self, msg: SyftMessage) -> bool:
        return (
            msg.address.pub_address.domain == self.id
            and msg.address.pri_address.device is None
        )

    @syft_decorator(typechecking=True)
    def get_client(self) -> DomainClient:
        conn_client = create_virtual_connection(node=self)
        return DomainClient(domain_id=self.id, name=self.name, connection=conn_client)
