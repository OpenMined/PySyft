from typing import List
from typing_extensions import final

from ....decorators import syft_decorator
from ...io.address import Address
from ...io.route import Route
from ..common.client import Client

from ...io.location import Location
from typing import Optional

@final
class VirtualMachineClient(Client):
    @syft_decorator(typechecking=True)
    def __init__(self,
                 name: str,
                 routes: List[Route],
                 network: Optional[Location] = None,
                 domain: Optional[Location] = None,
                 device: Optional[Location] = None,
                 vm: Optional[Location] = None):
        super().__init__(name=name,
                         routes=routes,
                         network=network,
                         domain=domain,
                         device=device,
                         vm=vm)

        # if this client doesn't know the ID of the VM it's supposed to point to
        # then something went wrong. The addressing system is a little fancy to
        # try to make sure that self.address is always up to date AND to work
        # with only one addressing system which is generic to all clients, so
        # I thought I'd add this here just as an extra check. It seems like an
        # ok thing to do since VMs shouldn't be spun up that often. Aka, VM
        # spinup time shouldn't be a huge constraint.
        assert self.vm is not None

    @property
    def id(self):
        return self.vm.id

    @syft_decorator(typechecking=True)
    def __repr__(self) -> str:
        out = f"<VirtualMachineClient id:{self.name}>"
        return out
