# stdlib
from threading import Thread
from time import sleep
from typing import Dict
from typing import Optional

# third party
from flask import current_app as app
import jwt
from nacl.signing import SigningKey
from nacl.signing import VerifyKey
import syft as sy
from syft import serialize
from syft.core.common.message import SignedImmediateSyftMessageWithReply
from syft.core.common.message import SignedImmediateSyftMessageWithoutReply
from syft.core.io.location import Location
from syft.core.io.location import SpecificLocation
from syft.core.node.common.action.exception_action import ExceptionMessage
from syft.core.node.common.action.exception_action import UnknownPrivateException
from syft.core.node.common.service.auth import AuthorizationException
from syft.core.node.device.client import DeviceClient
from syft.core.node.domain.domain import Domain
from syft.grid.connections.http_connection import HTTPConnection

# grid relative
from ..database import db
from ..database.store_disk import DiskObjectStore
from ..manager.association_request_manager import AssociationRequestManager
from ..manager.environment_manager import EnvironmentManager
from ..manager.group_manager import GroupManager
from ..manager.role_manager import RoleManager
from ..manager.setup_manager import SetupManager
from ..manager.user_manager import UserManager
from ..services.association_request import AssociationRequestService
from ..services.broadcast_search import BroadcastSearchService
from ..services.dataset_service import DatasetManagerService
from ..services.group_service import GroupManagerService
from ..services.infra_service import DomainInfrastructureService
from ..services.role_service import RoleManagerService
from ..services.setup_service import SetUpService
from ..services.tensor_service import RegisterTensorService
from ..services.transfer_service import TransferObjectService
from ..services.user_service import UserManagerService


class GridNetwork(Domain):
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
        self.disk_store = DiskObjectStore(db)
        self.setup = SetupManager(db)
        self.association_requests = AssociationRequestManager(db)

        self.env_clients = {}

        # Grid Domain Services
        self.immediate_services_with_reply.append(AssociationRequestService)
        self.immediate_services_with_reply.append(SetUpService)
        self.immediate_services_with_reply.append(RoleManagerService)
        self.immediate_services_with_reply.append(UserManagerService)
        self.immediate_services_with_reply.append(BroadcastSearchService)
        self._register_services()

    def login(self, email: str, password: str) -> Dict:
        user = self.users.login(email=email, password=password)
        token = jwt.encode({"id": user.id}, app.config["SECRET_KEY"])
        token = token.decode("UTF-8")
        return {
            "token": token,
            "key": user.private_key,
            "metadata": serialize(self.get_metadata_for_client())
            .SerializeToString()
            .decode("ISO-8859-1"),
        }

    def recv_immediate_msg_with_reply(
        self, msg: SignedImmediateSyftMessageWithReply, raise_exception=False
    ) -> SignedImmediateSyftMessageWithoutReply:
        if raise_exception:
            response = self.process_message(
                msg=msg, router=self.immediate_msg_with_reply_router
            )
            # maybe I shouldn't have created process_message because it screws up
            # all the type inference.
            res_msg = response.sign(signing_key=self.signing_key)  # type: ignore
        else:
            # exceptions can be easily triggered which break any WebRTC loops
            # so we need to catch them here and respond with a special exception
            # message reply
            try:
                # try to process message
                response = self.process_message(
                    msg=msg, router=self.immediate_msg_with_reply_router
                )

            except Exception as e:
                public_exception: Exception
                if isinstance(e, AuthorizationException):
                    private_log_msg = "An AuthorizationException has been triggered"
                    public_exception = e
                else:
                    private_log_msg = f"An {type(e)} has been triggered"  # dont send
                    public_exception = UnknownPrivateException(
                        "UnknownPrivateException has been triggered."
                    )
                try:
                    # try printing a useful message
                    private_log_msg += f" by {type(msg.message)} "
                    private_log_msg += f"from {msg.message.reply_to}"  # type: ignore
                except Exception:
                    pass

                # send the public exception back
                response = ExceptionMessage(
                    address=msg.message.reply_to,  # type: ignore
                    msg_id_causing_exception=msg.message.id,
                    exception_type=type(public_exception),
                    exception_msg=str(public_exception),
                )

            # maybe I shouldn't have created process_message because it screws up
            # all the type inference.
            res_msg = response.sign(signing_key=self.signing_key)  # type: ignore
            output = (
                f"> {self.pprint} Signing {res_msg.pprint} with "
                + f"{self.key_emoji(key=self.signing_key.verify_key)}"  # type: ignore
            )
        return res_msg
