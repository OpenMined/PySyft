from ..abstract.client import Client
from typing import final
from ....decorators import syft_decorator
from ....common.id import UID
from ...io.abstract import ClientConnection


@final
class VirtualMachineClient(Client):
    @syft_decorator(typechecking=True)
    def __init__(self, vm_id: UID, name: str, connection: ClientConnection):
        super().__init__(node_id=vm_id, name=name, connection=connection)

    @syft_decorator(typechecking=True)
    def __repr__(self) -> str:
        out = f"<VirtualMachineClient id:{self.name}>"
        return out
