# stdlib
from typing import List
from typing import Optional

# third party
from nacl.signing import SigningKey
from nacl.signing import VerifyKey
from typing_extensions import final

# relative
from ..common.serde.serializable import serializable
from ..common.uid import UID
from ..io.location import Location
from ..io.location.specific import SpecificLocation
from ..io.route import Route
from .common.client import Client


@final
@serializable(recursive_serde=True)
class VirtualMachineClient(Client):
    __attr_allowlist__ = [
        "name",
        "routes",
        "network",
        "domain",
        "device",
        "vm",
    ]

    vm: SpecificLocation  # redefine the type of self.vm to not be optional

    def __init__(
        self,
        node_uid: UID,
        name: Optional[str],
        routes: List[Route],
        vm: Optional[SpecificLocation] = None,
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
            node_uid=node_uid,
        )

        self.post_init()

    @property
    def id(self) -> UID:
        return self.node_uid

    @id.setter
    def id(self, new_id: UID) -> None:
        self.node_uid = new_id

    def __repr__(self) -> str:
        return f"<{type(self).__name__}: {self.name}>"
