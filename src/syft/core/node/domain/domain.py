from syft.core.common.message import SyftMessage

from ..common.node import Node
from ..device import Device, DeviceClient
from .client import DomainClient


class Domain(Node):

    client_type = DomainClient

    child_type = Device
    child_type_client_type = DeviceClient

    def __init__(self, name):
        super().__init__(name=name)
        # available_device_types = set()
        # TODO: add available compute types

        # default_device = None
        # TODO: add default compute type

        self._register_services()

    def add_me_to_my_address(self) -> None:
        self.address.pub_address.domain = self.id

    def message_is_for_me(self, msg: SyftMessage) -> bool:
        return (
            msg.address.pub_address.domain == self.id
            and msg.address.pri_address.device is None
        )
