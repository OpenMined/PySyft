from typing import Optional
from typing import Dict

from syft.core.node.domain.domain import Domain
from syft.core.node.device.client import DeviceClient
from syft.grid.connections.http_connection import HTTPConnection
from syft.core.io.location import SpecificLocation
from syft.core.io.location import Location

# Services
from .services.association_request import AssociationRequestService
from .services.infra_service import DomainInfrastructureService
from .services.setup_service import SetUpService
from .services.tensor_service import RegisterTensorService
from .services.role_service import RoleManagerService
from .services.user_service import UserManagerService

# Database Management
from .database import db
from .manager.user_manager import UserManager
from .manager.role_manager import RoleManager
from .manager.group_manager import GroupManager

from nacl.signing import SigningKey
from nacl.signing import VerifyKey

import jwt
from flask import current_app as app


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

        # Database Management Instances
        self.users = UserManager(db)
        self.roles = RoleManager(db)
        self.groups = GroupManager(db)

        # Grid Domain Services
        self.immediate_services_with_reply.append(AssociationRequestService)
        self.immediate_services_with_reply.append(DomainInfrastructureService)
        self.immediate_services_with_reply.append(SetUpService)
        self.immediate_services_with_reply.append(RegisterTensorService)
        self.immediate_services_with_reply.append(RoleManagerService)
        self.immediate_services_with_reply.append(UserManagerService)
        self._register_services()

    def login(self, email: str, password: str) -> Dict:
        user = self.users.login(email=email, password=password)
        token = jwt.encode({"id": user.id}, app.config["SECRET_KEY"])
        token = token.decode("UTF-8")
        return {
            "token": token,
            "key": user.private_key,
            "metadata": self.get_metadata_for_client()
            .serialize()
            .SerializeToString()
            .decode("ISO-8859-1"),
        }


node = GridDomain(name="om-domain")
