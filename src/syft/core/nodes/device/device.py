from ..abstract.worker import Worker
from typing import final
from ...io.virtual import create_virtual_connection
from .client import DeviceClient
from ....typecheck import type_hints
from ..vm.vm import VirtualMachine
from ..vm.client import VirtualMachineClient


@final
class Device(Worker):
    @type_hints
    def __init__(self, name: str):
        super().__init__(name=name)
        # the VM objects themselves
        self._vms = {}

        # clients to the VM objects
        self.vms = {}

        # a lookup table to lookup VMs by name instead of ID
        self.vm_name2id = {}

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

    @type_hints
    def _register_services(self) -> None:
        services = list()

        for s in services:
            self.msg_router[s.message_handler_type()] = s()
