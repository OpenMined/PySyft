from ..abstract.worker import Worker
from typing import final
from ...io.virtual import create_virtual_connection
from .client import DeviceClient
from ....typecheck import type_hints
from ..vm.vm import VirtualMachine
from ..vm.client import VirtualMachineClient
from .service.vm_service import VirtualMachineService
from ..common.device import AbstractDevice

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

    def create_vm(self, name: str) -> VirtualMachineClient:
        vm = VirtualMachine(name=name)

        client = vm.get_client()
        self.vms[vm.id] = client
        self.vm_name2id[vm.name] = vm.id

        return client

    @type_hints
    def get_client(self) -> DeviceClient:
        conn_client = create_virtual_connection(worker=self)
        return DeviceClient(device_id=self.id, connection=conn_client)


