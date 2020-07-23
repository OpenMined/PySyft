from .client import VirtualMachineClient
from ..abstract.node import Node
from . import service
from ...io.virtual import create_virtual_connection
from ....decorators import syft_decorator
from ...message import SyftMessage
from typing import final


@final
class VirtualMachine(Node):
    @syft_decorator(typechecking=True)
    def __init__(self, *args: list, **kwargs: str):
        super().__init__(*args, **kwargs)

        services = list()
        services.append(service.get_object_service.GetObjectService)
        services.append(service.save_object_service.SaveObjectService)
        services.append(service.run_class_service.RunClassMethodService)
        services.append(service.delete_object_service.DeleteObjectService)
        services.append(
            service.run_function_or_constructor_service.RunFunctionOrConstructorService
        )
        services.append(service.repr_service.ReprService)

        self._set_services(services=services)

    @syft_decorator(typechecking=True)
    def _recv_msg(self, msg: SyftMessage) -> SyftMessage:
        return self.recv_msg(msg=msg)

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
