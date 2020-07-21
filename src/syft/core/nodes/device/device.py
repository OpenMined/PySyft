from ..abstract.worker import Worker
from typing import final
from ...io.virtual import create_virtual_connection
from .client import DeviceClient
from ....decorators import syft_decorator
from .service.vm_msg_service import VirtualMachineMessageService
from .service.vm_lifecycle_service import VirtualMachineLifecycleService
from ..common.device import AbstractDevice
from ....common.id import UID


@final
class Device(Worker, AbstractDevice):
    @syft_decorator(typechecking=True)
    def __init__(self, name: str):
        super().__init__(name=name)

        remote_nodes = DeviceRemoteNodes()

        # the VM objects themselves
        self._vms = {}

        # clients to the VM objects
        self.vms = {}

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
        conn_client = create_virtual_connection(worker=self)
        return DeviceClient(device_id=self.id, name=self.name, connection=conn_client)
