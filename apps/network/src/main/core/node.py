from typing import Optional
from typing import Dict

from syft.core.common.message import SignedImmediateSyftMessageWithReply
from syft.core.common.message import SignedImmediateSyftMessageWithoutReply

from syft.core.node.common.action.exception_action import ExceptionMessage
from syft.core.node.common.action.exception_action import UnknownPrivateException
from syft.core.node.common.service.auth import AuthorizationException
from syft.core.node.network.network import Network
from syft.grid.connections.http_connection import HTTPConnection
from syft.core.io.location import SpecificLocation
from syft.core.io.location import Location

# Services
from .services.association_request import AssociationRequestService
from .services.setup_service import SetUpService
from .services.role_service import RoleManagerService
from .services.user_service import UserManagerService

# Database Management
from .database import db
from .manager.user_manager import UserManager
from .manager.role_manager import RoleManager
from .manager.setup_manager import SetupManager

from nacl.signing import SigningKey
from nacl.signing import VerifyKey
from time import sleep

import jwt
from flask import current_app as app
from threading import Thread

import syft as sy


class GridNetwork(Network):
    def __init__(
        self,
        name: Optional[str],
        network: Optional[Location] = SpecificLocation(),
        domain: SpecificLocation = None,
        device: Optional[Location] = None,
        vm: Optional[Location] = None,
        signing_key: Optional[SigningKey] = None,
        verify_key: Optional[VerifyKey] = None,
        root_key: Optional[VerifyKey] = None,
    ):
        super().__init__(
            name=name,
            network=network,
            domain=domain,
            device=device,
            vm=vm,
            signing_key=signing_key,
            verify_key=verify_key,
        )

        # Database Management Instances
        self.users = UserManager(db)
        self.roles = RoleManager(db)
        self.setup = SetupManager(db)

        # Grid Domain Services
        self.immediate_services_with_reply.append(AssociationRequestService)
        self.immediate_services_with_reply.append(SetUpService)
        self.immediate_services_with_reply.append(RoleManagerService)
        self.immediate_services_with_reply.append(UserManagerService)
        self._register_services()

        thread = Thread(target=self.thread_run_handlers)
        thread.start()

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

    def thread_run_handlers(self) -> None:
        while True:
            sleep(0.1)
            try:
                self.clean_up_handlers()
                self.clean_up_requests()
                if len(self.request_handlers) > 0:
                    for request in self.requests:
                        # check if we have previously already handled this in an earlier iter
                        if request.id not in self.handled_requests:
                            for handler in self.request_handlers:
                                handled = self.check_handler(
                                    handler=handler, request=request
                                )
                                if handled:
                                    # we handled the request so we can exit the loop
                                    break
            except Exception as excp2:
                print(str(excp2))


node = GridNetwork(name="om-net")
