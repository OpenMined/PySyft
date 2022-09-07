# future
from __future__ import annotations

# stdlib
import asyncio
import os
import time
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

# third party
import ascii_magic
from nacl.signing import SigningKey
from nacl.signing import VerifyKey
from pydantic import BaseSettings

# relative
from ...lib.python import String
from ...logger import critical
from ...logger import debug
from ...logger import info
from ...logger import traceback
from ..adp.ledger_store import RedisLedgerStore
from ..common.message import SignedImmediateSyftMessageWithReply
from ..common.message import SignedMessage
from ..common.message import SyftMessage
from ..common.uid import UID
from ..io.location import Location
from ..io.location import SpecificLocation
from .common.action.get_object_action import GetObjectAction
from .common.client import Client
from .common.node import Node
from .common.node_manager.association_request_manager import AssociationRequestManager
from .common.node_manager.dataset_manager import DatasetManager
from .common.node_manager.environment_manager import EnvironmentManager
from .common.node_manager.node_manager import NodeManager
from .common.node_manager.node_route_manager import NodeRouteManager
from .common.node_manager.redis_store import RedisStore
from .common.node_manager.request_manager import RequestManager
from .common.node_manager.role_manager import RoleManager
from .common.node_manager.user_manager import UserManager
from .common.node_service.association_request.association_request_service import (
    AssociationRequestService,
)
from .common.node_service.association_request.association_request_service import (
    AssociationRequestWithoutReplyService,
)
from .common.node_service.dataset_manager.dataset_manager_service import (
    DatasetManagerService,
)
from .common.node_service.get_remaining_budget.get_remaining_budget_service import (
    GetRemainingBudgetService,
)
from .common.node_service.node_setup.node_setup_messages import (
    CreateInitialSetUpMessage,
)
from .common.node_service.node_setup.node_setup_service import NodeSetupService
from .common.node_service.object_request.object_request_service import (
    ObjectRequestServiceWithoutReply,
)
from .common.node_service.object_request.object_request_service import RequestService
from .common.node_service.ping.ping_service import PingService
from .common.node_service.publish.publish_service import PublishScalarsService
from .common.node_service.request_answer.request_answer_messages import RequestStatus
from .common.node_service.request_answer.request_answer_service import (
    RequestAnswerService,
)
from .common.node_service.request_receiver.request_receiver_messages import (
    RequestMessage,
)
from .common.node_service.role_manager.role_manager_service import RoleManagerService
from .common.node_service.simple.simple_service import SimpleService
from .common.node_service.sleep.sleep_service import SleepService
from .common.node_service.user_auth.user_auth_service import UserLoginService
from .common.node_service.user_manager.user_manager_service import UserManagerService
from .common.node_service.vpn.vpn_service import VPNConnectService
from .common.node_service.vpn.vpn_service import VPNJoinService
from .common.node_service.vpn.vpn_service import VPNStatusService
from .common.node_table.utils import create_memory_db_engine
from .device import Device
from .device import DeviceClient
from .domain_client import DomainClient
from .domain_service import DomainServiceClass


class Domain(Node):
    domain: SpecificLocation
    root_key: Optional[VerifyKey]

    child_type = Device
    client_type = DomainClient
    child_type_client_type = DeviceClient

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
        db_engine: Any = None,
        store_type: type = RedisStore,
        ledger_store_type: type = RedisLedgerStore,
        settings: Optional[BaseSettings] = None,
    ):

        if db_engine is None:
            db_engine, _ = create_memory_db_engine()

        super().__init__(
            name=name,
            network=network,
            domain=domain,
            device=device,
            vm=vm,
            signing_key=signing_key,
            verify_key=verify_key,
            db_engine=db_engine,
            store_type=store_type,
            settings=settings,
        )

        # share settings with the FastAPI application level
        self.settings = settings

        # specific location with name
        self.domain = SpecificLocation(name=self.name)
        self.root_key = root_key

        # Database Management Instances
        self.users = UserManager(db_engine)
        self.roles = RoleManager(db_engine)
        self.environments = EnvironmentManager(db_engine)
        self.association_requests = AssociationRequestManager(db_engine)
        self.data_requests = RequestManager(db_engine)
        self.datasets = DatasetManager(db_engine)
        self.node = NodeManager(db_engine)
        self.node_route = NodeRouteManager(db_engine)
        self.ledger_store = ledger_store_type(settings=settings)

        # self.immediate_services_without_reply.append(RequestReceiverService)
        # self.immediate_services_without_reply.append(AcceptOrDenyRequestService)
        # self.immediate_services_without_reply.append(UpdateRequestHandlerService)
        self.immediate_services_without_reply.append(PublishScalarsService)
        self.immediate_services_with_reply.append(RequestAnswerService)
        # self.immediate_services_with_reply.append(GetAllRequestHandlersService)

        # Grid Domain Services
        self.immediate_services_with_reply.append(AssociationRequestService)
        # self.immediate_services_with_reply.append(DomainInfrastructureService)
        self.immediate_services_with_reply.append(GetRemainingBudgetService)
        self.immediate_services_with_reply.append(SimpleService)
        self.immediate_services_with_reply.append(PingService)
        self.immediate_services_with_reply.append(VPNConnectService)
        self.immediate_services_with_reply.append(VPNJoinService)
        self.immediate_services_with_reply.append(VPNStatusService)
        self.immediate_services_with_reply.append(NodeSetupService)
        self.immediate_services_with_reply.append(RoleManagerService)
        self.immediate_services_with_reply.append(UserManagerService)
        self.immediate_services_with_reply.append(DatasetManagerService)
        self.immediate_services_with_reply.append(RequestService)
        self.immediate_services_with_reply.append(UserLoginService)

        self.immediate_services_without_reply.append(ObjectRequestServiceWithoutReply)
        self.immediate_services_without_reply.append(
            AssociationRequestWithoutReplyService
        )

        # TODO: New Service registration process
        self.immediate_services_with_reply.append(DomainServiceClass)

        # TODO: @Madhava change to a map of accountants that are created on first
        # use of the DS key

        if getattr(self.settings, "TEST_MODE", False):
            print("Loading TEST_MODE services")
            # only add in test mode
            self.immediate_services_with_reply.append(SleepService)

        self.requests: List[RequestMessage] = list()
        # available_device_types = set()
        # TODO: add available compute types

        # default_device = None
        # TODO: add default compute type

        self._register_services()
        self.request_handlers: List[Dict[Union[str, String], Any]] = []
        self.handled_requests: Dict[Any, float] = {}

        self.post_init()

        # run the handlers in an asyncio future
        asyncio.ensure_future(self.run_handlers())

    def post_init(self) -> None:
        super().post_init()
        self.set_node_uid()
        if not hasattr(self, "signing_key"):
            Node.set_keys(node=self)

    def initial_setup(  # nosec
        self,
        signing_key: SigningKey,
        first_superuser_name: str = "Jane Doe",
        first_superuser_email: str = "info@openmined.org",
        first_superuser_password: str = "changethis",
        first_superuser_budget: float = 5.55,
        domain_name: str = "BigHospital",
    ) -> Domain:
        Node.set_keys(node=self, signing_key=signing_key)

        # Build Syft Message
        msg: SignedImmediateSyftMessageWithReply = CreateInitialSetUpMessage(
            address=self.address,
            name=first_superuser_name,
            email=first_superuser_email,
            password=first_superuser_password,
            domain_name=domain_name,
            budget=first_superuser_budget,
            reply_to=self.address,
            signing_key=signing_key,
        ).sign(signing_key=self.signing_key)

        # Process syft message
        _ = self.recv_immediate_msg_with_reply(msg=msg).message

        return self

    def loud_print(self) -> None:
        try:
            install_path = os.path.abspath(
                os.path.join(os.path.realpath(__file__), "../../../img/")
            )
            ascii_magic.to_terminal(
                ascii_magic.from_image_file(
                    img_path=install_path + "/pygrid.png", columns=83
                )
            )

            print(
                r"""
                                                         __
                                                        |  \  _   _   _  .  _
                                                        |__/ (_) ||| (_| | | )
    """
            )
        except Exception:
            print("DOMAIN (broken file path)")

    @property
    def icon(self) -> str:
        return "🏰"

    @property
    def id(self) -> UID:
        return self.domain.id

    def message_is_for_me(self, msg: Union[SyftMessage, SignedMessage]) -> bool:

        # this needs to be defensive by checking domain_id NOT domain.id or it breaks
        try:
            return msg.address.domain_id == self.id and msg.address.device is None
        except Exception as excp3:
            critical(
                f"Error checking if {msg.pprint} is for me on {self.pprint}. {excp3}"
            )
            return False

    def set_request_status(
        self, message_request_id: UID, status: RequestStatus, client: Client
    ) -> bool:
        for req in self.requests:
            if req.request_id == message_request_id:
                req.owner_client_if_available = client
                if status == RequestStatus.Accepted:
                    req.accept()
                    return True
                elif status == RequestStatus.Rejected:
                    req.deny()
                    return True

        return False

    def get_request_status(self, message_request_id: UID) -> RequestStatus:
        # is it still pending
        for req in self.requests:
            if req.request_id == message_request_id:
                return RequestStatus.Pending

        # check if it was accepted
        # TODO remove brute search of all store objects
        # Currently theres no way to find which object to check the permissions
        # to find the stored request_id
        for obj_id in self.store.keys():
            for _, request_id in self.store.get(
                obj_id, proxy_only=True
            ).read_permissions.items():
                if request_id == message_request_id:
                    return RequestStatus.Accepted

            for _, request_id in self.store.get(
                obj_id, proxy_only=True
            ).search_permissions.items():
                if request_id == message_request_id:
                    return RequestStatus.Accepted

        # must have been rejected
        return RequestStatus.Rejected

    def _get_object(self, request: RequestMessage) -> Optional[Any]:
        try:
            obj_msg = GetObjectAction(
                id_at_location=request.object_id,
                address=request.owner_address,
                reply_to=self.address,
                delete_obj=False,
            )

            service = self.immediate_msg_with_reply_router[type(obj_msg)]
            response = service.process(
                node=self, msg=obj_msg, verify_key=self.root_verify_key
            )
            if response:
                obj = getattr(response, "obj", None)
                if obj is not None:
                    return obj
        except Exception as excp1:
            critical(f"Exception getting object for {request}. {excp1}")
        return None

    def _count_elements(self, obj: object) -> Tuple[bool, int]:
        allowed = False
        elements = 0

        nelement = getattr(obj, "nelement", None)
        if nelement is not None:
            elements = max(elements, int(nelement()))
            allowed = True

        return (allowed, elements)

    def _accept(self, request: RequestMessage) -> None:
        debug(f"Calling accept on request: {request.id}")
        request.destination_node_if_available = self
        request.accept()

    def _deny(self, request: RequestMessage) -> None:
        debug(f"Calling deny on request: {request.id}")
        request.destination_node_if_available = self
        request.deny()

    def _try_deduct_quota(
        self, handler: Dict[Union[str, String], Any], obj: Any
    ) -> bool:
        action = handler.get("action", None)
        if action == "accept":
            allowed, element_count = self._count_elements(obj=obj)
            if allowed:
                result = handler["element_quota"] - element_count
                if result >= 0:
                    # the request will be accepted so lets decrement the quota
                    handler["element_quota"] = max(0, result)
                    return True

        return False

    def check_handler(
        self, handler: Dict[Union[str, String], Any], request: RequestMessage
    ) -> bool:
        debug(f"HANDLER Check handler {handler} against {request.request_id}")

        tags = handler.get("tags", [])

        action = handler.get("action", None)
        print_local = handler.get("print_local", None)
        log_local = handler.get("log_local", None)
        element_quota = handler.get("element_quota", None)

        # We match a handler and a request when they have a same set of tags,
        # or if handler["tags"]=[], it matches with any request.
        if len(tags) > 0 and not set(request.object_tags) == set(tags):
            debug(f"HANDLER Ignoring request handler {handler} against {request}")
            return False

        # if we have any of these three rules we will need to get the object to
        # print it, log it, or check its quota
        obj = None
        if print_local or log_local or element_quota:
            obj = self._get_object(request=request)
            debug(f"> HANDLER Got object {obj} for checking")

        # we only want to accept or deny once
        handled = False

        # check quota and reject first
        if element_quota is not None:
            if not self._try_deduct_quota(handler=handler, obj=obj):
                debug(
                    f"> HANDLER Rejecting {request} element_quota={handler['element_quota']}"
                )
                self._deny(request=request)
                handled = True

        # if not rejected based on quota keep checking
        if not handled:
            if action == "accept":
                debug(f"Check accept {handler} against {request}")
                self._accept(request=request)
                handled = True
            elif action == "deny":
                self._deny(request=request)
                handled = True

        # print or log rules can execute multiple times so no complex logic here
        if print_local or log_local:
            log = f"> HANDLER Request {request.request_id}:"
            if len(request.request_description) > 0:
                log += f" {request.request_description}"
            log += f"\nValue: {obj}"

            # if these are enabled output them
            if print_local:
                critical(log)

            if log_local:
                info(log)

        # block the loop from handling this again, until the cleanup removes it
        # after a period of timeout
        if handled:
            self.handled_requests[request.id] = time.time()
        return handled

    def clean_up_handlers(self) -> None:
        # this makes sure handlers with timeout expire
        now = time.time()
        alive_handlers = []
        if len(self.request_handlers) > 0:
            for handler in self.request_handlers:
                timeout_secs = handler.get("timeout_secs", -1)
                if timeout_secs != -1:
                    created_time = handler.get("created_time", 0)
                    if now - created_time > timeout_secs:
                        continue
                alive_handlers.append(handler)
        self.request_handlers = alive_handlers

    def clear(self, user_role: int) -> bool:
        # Cleanup database tables
        if user_role == self.roles.owner_role.id:
            self.store.clear()
            self.data_requests.clear()
            self.users.clear()
            self.environments.clear()
            self.association_requests.clear()
            self.datasets.clear()
            self.initial_setup(signing_key=self.signing_key)
            return True

        return False

    def clean_up_requests(self) -> None:
        # this allows a request to be re-handled if the handler somehow failed
        now = time.time()
        processing_wait_secs = 5
        reqs_to_remove = []
        for req in self.handled_requests.keys():
            handle_time = self.handled_requests[req]
            if now - handle_time > processing_wait_secs:
                reqs_to_remove.append(req)

        for req in reqs_to_remove:
            self.handled_requests.__delitem__(req)

        alive_requests: List[RequestMessage] = []
        for request in self.requests:
            if request.timeout_secs is not None and request.timeout_secs > -1:
                if request.arrival_time is None:
                    critical(f"HANDLER Request has no arrival time. {request.id}")
                    request.set_arrival_time(arrival_time=time.time())
                arrival_time = getattr(request, "arrival_time", float(now))
                if now - arrival_time > request.timeout_secs:
                    # this request has expired
                    continue
            alive_requests.append(request)

        self.requests = alive_requests

    async def run_handlers(self) -> None:
        while True:
            await asyncio.sleep(0.01)
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
                traceback(excp2)
