from ..abstract.client import Client
from typing import final
from ....decorators import syft_decorator
from ....common.id import UID
from ...io.abstract import ClientConnection


@final
class VirtualMachineClient(Client):
    @syft_decorator(typechecking=True)
    def __init__(self, vm_id: UID, name: str, connection: ClientConnection):
        super().__init__(target_node_id=vm_id, name=name, connection=connection)

    def add_me_to_my_address(self):
        self.address.pri_address.vm = self.vm_id

    @property
    def target_node_id(self) -> UID:
        """This client points to a vm, this returns the id of that vm."""
        return self.vm_id

    @target_node_id.setter
    def target_node_id(self, new_target_node_id: UID) -> UID:
        """This client points to a vm, this saves the id of that vm"""
        self.vm_id = new_target_node_id
        return self.vm_id

    @syft_decorator(typechecking=True)
    def __repr__(self) -> str:
        out = f"<VirtualMachineClient id:{self.name}>"
        return out
