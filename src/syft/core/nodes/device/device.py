from ..abstract.worker import Worker
from typing import final
from ...io.virtual import create_virtual_connection
from .client import DeviceClient
from ....typecheck import type_hints
from ..vm.vm import VirtualMachine
from ..vm.client import VirtualMachineClient
from .service.vm_service import VirtualMachineService
from ..common.device import AbstractDevice
from ....common.id import UID


@final
class Device(Worker, AbstractDevice):
    @type_hints
    def __init__(self, name: str):
        super().__init__(name=name)

        # the VM objects themselves
        self._vms = {}

        # clients to the VM objects
        self.vms = {}

        # a lookup table to lookup VMs by name instead of ID
        self.vm_name2id = {}

        services = list()
        # TODO: populate list of services
        services.append(VirtualMachineService)
        self._set_services(services=services)

    def get_vm(self, id: UID = None, name: str = None):
        if id is not None:
            return self.vms[id]
        elif name is not None:
            id = self.vm_name2id[name]
            return self.vms[id]
        else:
            raise Exception("You must ask for a vm using either a name or ID.")

    def create_vm(self, name: str) -> VirtualMachineClient:
        vm = VirtualMachine(name=name)

        client = vm.get_client()
        self.vms[vm.id] = client
        self.vm_name2id[vm.name] = vm.id

        return client

    @type_hints
    def get_client(self) -> DeviceClient:
        conn_client = create_virtual_connection(worker=self)
        return DeviceClient(device_id=self.id, name=self.name, connection=conn_client)
