# external classs imports
from typing import Optional

# external decorators
from typing_extensions import final

# syft imports
from ...io.location import SpecificLocation
from ...common.message import SyftMessage
from ....decorators import syft_decorator
from .client import VirtualMachineClient
from ...io.location import Location
from ..common.node import Node
from ...common.uid import UID


@final
class VirtualMachine(Node):

    client_type = VirtualMachineClient
    vm: SpecificLocation  # redefine the type of self.vm to not be optional

    @syft_decorator(typechecking=True)
    def __init__(
        self,
        name: str,
        network: Optional[Location] = None,
        domain: Optional[Location] = None,
        device: Optional[Location] = None,
        vm: SpecificLocation = SpecificLocation(),
    ):
        super().__init__(
            name=name, network=network, domain=domain, device=device, vm=vm
        )

        # All node subclasses have to call this at the end of their __init__
        self._register_services()

    @property
    def id(self) -> UID:
        return self.vm.id

    def message_is_for_me(self, msg: SyftMessage) -> bool:
        return msg.address.vm.id == self.id

    @syft_decorator(typechecking=True)
    def _register_frameworks(self) -> None:
        raise NotImplementedError
        # TODO: it doesn't at the moment but it needs to in the future,
        #  mostly because nodes should be able to choose waht framweorks they
        #  want to support (and more importantly what versions of those frameworks
        #  they want to support).
        # QUESTION: Does this exist?
        # from ....lib import supported_frameworks
        # for fw in supported_frameworks:
        #     for name, ast in fw.ast.attrs.items():
        #         if name in self.frameworks.attrs:
        #             raise KeyError(
        #                 "Framework already imported. Why are you importing it twice?"
        #             )
        #         self.frameworks.attrs[name] = ast
