from ..abstract.client import Client
from typing import final
from ....typecheck import type_hints
from ....common.id import UID
from ...io.abstract import ClientConnection


@final
class VirtualMachineClient(Client):
    @type_hints
    def __init__(self, vm_id: UID, connection: ClientConnection):
        super().__init__(worker_id=vm_id, connection=connection)

    @type_hints
    def __repr__(self) -> str:
        out = f"<VirtualMachineClient id:{self.name}>"
        out += "\n" + str(self.vm.store._objects)  # TODO: remove this - security risk
        return out
