from ..abstract.node import Node
from typing import final
from ...io.virtual import create_virtual_connection
from .client import DeviceClient
from ....decorators import syft_decorator
from .service.vm_msg_service import VirtualMachineMessageService
from .service.vm_lifecycle_service import VirtualMachineLifecycleService
from ..common.device import AbstractDevice
from ....common.id import UID
from .device_type.device_type import DeviceType
from .device_type.unknown import unknown_device
from ..vm.vm import VirtualMachine
from typing import Dict

@final
class Device(Node, AbstractDevice):
    @syft_decorator(typechecking=True)
    def __init__(self, name: str, device_type: DeviceType=unknown_device, vms: Dict[UID, VirtualMachine]={}):
        super().__init__(name=name)

        # the VM objects themselves
        self._vms = vms

        # clients to the VM objects
        self.vms = {}
        for key, vm in self._vms.items():
            self.vms[key] = vm.get_client()


        # a lookup table to lookup VMs by name instead of ID
        self.vm_name2id = {}

        services = list()
        services.append(VirtualMachineMessageService)
        services.append(VirtualMachineLifecycleService)
        self._set_services(services=services)

    def get_vm(self, id_or_name:(str, UID)):
        """
        Fetch this from the network details.
        """
        try:
            return self.vms[id_or_name]
        except KeyError as e:
            try:
                id = self.vm_name2id[id_or_name]
                return self.vms[id]
            except KeyError as e:
                raise KeyError("You must ask for a vm using either a name or ID.")

    @syft_decorator(typechecking=True)
    def get_client(self) -> DeviceClient:
        conn_client = create_virtual_connection(node=self)
        return DeviceClient(device_id=self.id, name=self.name, connection=conn_client)
