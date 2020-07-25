from .client import VirtualMachineClient
from ..abstract.node import Node
from . import service
from ...io.virtual import create_virtual_connection
from ....decorators import syft_decorator
from ...message.syft_message import SyftMessage
from typing import final


@final
class VirtualMachine(Node):
    @syft_decorator(typechecking=True)
    def __init__(self, *args: list, **kwargs: str):
        super().__init__(*args, **kwargs)

        self._register_services()

    def message_is_for_me(self, msg:SyftMessage) -> bool:
        return msg.address.pri_address.vm == self.id

    @syft_decorator(typechecking=True)
    def get_client(self) -> VirtualMachineClient:
        conn = create_virtual_connection(node=self)
        return VirtualMachineClient(vm_id=self.id, name=self.name, connection=conn)

    @syft_decorator(typechecking=True)
    def _register_frameworks(self) -> None:

        from ....lib import supported_frameworks

        for fw in supported_frameworks:
            for name, ast in fw.ast.attrs.items():
                if name in self.frameworks.attrs:
                    raise KeyError(
                        "Framework already imported. Why are you importing it twice?"
                    )
                self.frameworks.attrs[name] = ast

    def __repr__(self):
        return f"VirtualMachine:{self.name}"
