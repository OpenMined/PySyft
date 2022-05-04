# stdlib
from typing import List
from typing import Optional

# third party
from nacl.signing import SigningKey
from nacl.signing import VerifyKey
from typing_extensions import final

# relative
from ..common.uid import UID
from ..io.location import Location
from ..io.location.specific import SpecificLocation
from ..io.route import Route
from .common.client import Client


@final
class VirtualMachineClient(Client):

    vm: SpecificLocation  # redefine the type of self.vm to not be optional

    def __init__(
        self,
        name: Optional[str],
        routes: List[Route],
        vm: SpecificLocation,
        network: Optional[Location] = None,
        domain: Optional[Location] = None,
        device: Optional[Location] = None,
        signing_key: Optional[SigningKey] = None,
        verify_key: Optional[VerifyKey] = None,
    ):
        super().__init__(
            name=name,
            routes=routes,
            network=network,
            domain=domain,
            device=device,
            vm=vm,
            signing_key=signing_key,
            verify_key=verify_key,
        )

        self.post_init()

    @property
    def id(self) -> UID:
        return self.vm.id

    def __repr__(self) -> str:
        return f"<{type(self).__name__}: {self.name}>"
