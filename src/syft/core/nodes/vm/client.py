from ..abstract.client import Client
from typing import final
from ....decorators import syft_decorator
from ...io.address import Address
from ...io.abstract import ClientConnection


@final
class VirtualMachineClient(Client):
    @syft_decorator(typechecking=True)
    def __init__(self, address: Address, name: str, connection: ClientConnection):
        super().__init__(address=address, name=name, connection=connection)

        # if this client doesn't know the ID of the VM it's supposed to point to
        # then something went wrong. The addressing system is a little fancy to
        # try to make sure that self.address is always up to date AND to work
        # with only one addressing system which is generic to all clients, so
        # I thought I'd add this here just as an extra check. It seems like an
        # ok thing to do since VMs shouldn't be spun up that often. Aka, VM
        # spinup time shouldn't be a huge constraint.
        assert self.vm_id is not None

    def add_me_to_my_address(self):
        assert self.vm_id is not None

    # @property
    # def target_node_id(self) -> UID:
    #     """This client points to a vm, this returns the id of that vm."""
    #     return self.vm_id
    #
    # @target_node_id.setter
    # def target_node_id(self, new_target_node_id: UID) -> UID:
    #     """This client points to a vm, this saves the id of that vm"""
    #     self.vm_id = new_target_node_id
    #     return self.vm_id

    @syft_decorator(typechecking=True)
    def __repr__(self) -> str:
        out = f"<VirtualMachineClient id:{self.name}>"
        return out
