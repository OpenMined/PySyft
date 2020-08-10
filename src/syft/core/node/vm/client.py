# external class imports
from typing import List
from typing import Optional
from typing_extensions import final

# syft imports
from ...io.location.specific import SpecificLocation
from ....decorators import syft_decorator
from ...io.location import Location
from ..common.client import Client
from ...io.route import Route
from ...common.uid import UID


@final
class VirtualMachineClient(Client):

    vm: SpecificLocation  # redefine the type of self.vm to not be optional

    @syft_decorator(typechecking=True)
    def __init__(
        self,
        name: str,
        routes: List[Route],
        vm: SpecificLocation,
        network: Optional[Location] = None,
        domain: Optional[Location] = None,
        device: Optional[Location] = None,
    ):
        super().__init__(
            name=name,
            routes=routes,
            network=network,
            domain=domain,
            device=device,
            vm=vm,
        )

    @property
    def id(self) -> UID:
        return self.vm.id

    @syft_decorator(typechecking=True)
    def __repr__(self) -> str:
        out = f"<VirtualMachineClient id:{self.name}>"
        return out
