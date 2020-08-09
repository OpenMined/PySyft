from typing_extensions import final
from typing import Optional

from syft.core.common.message import SyftMessage

from ....decorators import syft_decorator
from ..common.node import Node
from .client import VirtualMachineClient
from ...io.location import Location
from ...io.location import SpecificLocation


@final
class VirtualMachine(Node):

    client_type = VirtualMachineClient

    @syft_decorator(typechecking=True)
    def __init__(
        self,
        name: str,
        network: Optional[Location] = None,
        domain: Optional[Location] = None,
        device: Optional[Location] = None,
        vm: Optional[SpecificLocation] = SpecificLocation(),
    ):
        super().__init__(
            name=name, network=network, domain=domain, device=device, vm=vm
        )

        # All node subclasses have to call this at the end of their __init__
        self._register_services()

        # if this VM doesn't even know the id of itself at this point
        # then something went wrong with the fancy address system.
        assert self.vm is not None

    @property
    def id(self):
        return self.vm.id

    def message_is_for_me(self, msg: SyftMessage) -> bool:
        return msg.address.vm.id == self.id

    @syft_decorator(typechecking=True)
    def _register_frameworks(self) -> None:
        raise NotImplementedError
        # QUESTION: Does this exist?
        # from ....lib import supported_frameworks
        # for fw in supported_frameworks:
        #     for name, ast in fw.ast.attrs.items():
        #         if name in self.frameworks.attrs:
        #             raise KeyError(
        #                 "Framework already imported. Why are you importing it twice?"
        #             )
        #         self.frameworks.attrs[name] = ast
