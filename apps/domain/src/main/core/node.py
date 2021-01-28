from typing import Optional

# from syft.grid.grid_client import connect
from syft.core.node.domain.domain import Domain
from syft.core.node.device.client import DeviceClient
from syft.grid.connections.http_connection import HTTPConnection
from syft.core.io.location import SpecificLocation
from syft.core.io.location import Location
from .services.association_request import AssociationRequestService
from .services.infra_service import DomainInfrastructureService
from .services.setup_service import SetUpService
from .services.tensor_service import RegisterTensorService
from .services.role_service import RoleManagerService

from nacl.signing import SigningKey
from nacl.signing import VerifyKey


class GridDomain(Domain):
    def __init__(
        self,
        name: Optional[str],
        network: Optional[Location] = None,
        domain: SpecificLocation = SpecificLocation(),
        device: Optional[Location] = None,
        vm: Optional[Location] = None,
        signing_key: Optional[SigningKey] = None,
        verify_key: Optional[VerifyKey] = None,
        root_key: Optional[VerifyKey] = None,
        db_path: Optional[str] = None,
    ):
        super().__init__(
            name=name,
            network=network,
            domain=domain,
            device=device,
            vm=vm,
            signing_key=signing_key,
            verify_key=verify_key,
            db_path=db_path,
        )

        # Grid Domain Services
        self.immediate_services_with_reply.append(AssociationRequestService)
        self.immediate_services_with_reply.append(DomainInfrastructureService)
        self.immediate_services_with_reply.append(SetUpService)
        self.immediate_services_with_reply.append(RegisterTensorService)
        self.immediate_services_with_reply.append(RoleManagerService)
        self._register_services()


node = GridDomain(name="om-domain")
