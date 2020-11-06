# stdlib
import asyncio
import time
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

# third party
from loguru import logger
from nacl.signing import SigningKey
from nacl.signing import VerifyKey

# syft relative
from ....decorators.syft_decorator_impl import syft_decorator
from ....lib.python import String
from ...common.message import SignedMessage
from ...common.message import SyftMessage
from ...common.uid import UID
from ...io.location import Location
from ...io.location import SpecificLocation
from ..common.action.get_object_action import GetObjectAction
from ..common.client import Client
from ..common.node import Node
from ..device import Device
from ..device import DeviceClient
from .client import DomainClient
from .service import RequestAnswerMessageService
from .service import RequestMessage
from .service import RequestService
from .service import RequestStatus
from .service.accept_or_deny_request_service import AcceptOrDenyRequestService
from .service.get_all_requests_service import GetAllRequestsService
from .service.request_handler_service import GetAllRequestHandlersService
from .service.request_handler_service import UpdateRequestHandlerService


class Domain(Node):
    domain: SpecificLocation
    root_key: Optional[VerifyKey]

    child_type = Device
    client_type = DomainClient
    child_type_client_type = DeviceClient

    @syft_decorator(typechecking=True)
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
        # specific location with name
        self.domain = SpecificLocation(name=self.name)
        self.root_key = root_key

        self.immediate_services_without_reply.append(RequestService)
        self.immediate_services_without_reply.append(AcceptOrDenyRequestService)
        self.immediate_services_without_reply.append(UpdateRequestHandlerService)

        self.immediate_services_with_reply.append(RequestAnswerMessageService)
        self.immediate_services_with_reply.append(GetAllRequestsService)
        self.immediate_services_with_reply.append(GetAllRequestHandlersService)

        self.requests: List[RequestMessage] = list()
        # available_device_types = set()
        # TODO: add available compute types

        # default_device = None
        # TODO: add default compute type

        self._register_services()
        self.request_handlers: List[Dict[str, Any]] = []
        self.handled_requests: Dict[Any, float] = {}

        self.post_init()

        # run the handlers in an asyncio future
        asyncio.ensure_future(self.run_handlers())

    @property
    def icon(self) -> str:
        return "🏰"

    @property
    def id(self) -> UID:
        return self.domain.id

    @syft_decorator(typechecking=True)
    def message_is_for_me(self, msg: Union[SyftMessage, SignedMessage]) -> bool:

        # this needs to be defensive by checking domain_id NOT domain.id or it breaks
        try:
            return msg.address.domain_id == self.id and msg.address.device is None
        except Exception as e:
            error = f"Error checking if {msg.pprint} is for me on {self.pprint}. {e}"
            print(error)
            return False

    @syft_decorator(typechecking=True)
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

    @syft_decorator(typechecking=True)
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
            for _, request_id in self.store[obj_id].read_permissions.items():
                if request_id == message_request_id:
                    return RequestStatus.Accepted

            for _, request_id in self.store[obj_id].search_permissions.items():
                if request_id == message_request_id:
                    return RequestStatus.Accepted

        # must have been rejected
        return RequestStatus.Rejected

    @syft_decorator(typechecking=True)
    def check_handler(
        self, handler: Dict[Union[str, String], Any], request: RequestMessage
    ) -> bool:
        logger.debug(f"Check handler {handler} against {request}")
        if (
            "name" in handler
            and handler["name"] != ""
            and handler["name"] != request.name
        ):
            # valid name doesnt match so ignore this handler
            logger.debug(f"Ignoring request handler {handler} against {request}")
            return False

        obj = None

        # TODO: refactor this horrid mess
        # we only want to accept or deny once
        handled = False
        if "action" in handler:
            action = handler["action"]
            if action == "accept":
                logger.debug(f"Check accept {handler} against {request}")
                element_quota = 0
                if "element_quota" in handler:
                    element_quota = handler["element_quota"]
                logger.debug(
                    f"Check handler element quota {element_quota} against {request}"
                )
                if element_quota > 0:
                    try:
                        if obj is None:
                            obj_msg = GetObjectAction(
                                id_at_location=request.object_id,
                                address=request.owner_address,
                                reply_to=self.address,
                                delete_obj=False,
                            )

                            service = self.immediate_msg_with_reply_router[
                                type(obj_msg)
                            ]
                            response = service.process(
                                node=self, msg=obj_msg, verify_key=self.root_verify_key
                            )
                            obj = response.obj
                    except Exception as e:
                        logger.critical(f"Exception getting object. {e}")

                    elements = 0
                    nelement = getattr(obj, "nelement", None)
                    if nelement is not None:
                        print(nelement)
                        elements = int(nelement())
                        print("elmenets ype", type(elements))
                        if elements < 1:
                            length = getattr(obj, "__len__", None)
                            if length is not None:
                                elements = int(length())
                                print("length type", type(elements))
                    elements = max(1, elements)

                    remaining = element_quota - elements
                    if remaining >= 0:
                        print("handler before", handler)
                        handler["element_quota"] = remaining  # save?
                        print("handler after save", handler)
                        logger.debug(f"Calling accept on request: {request.id}")
                        request.destination_node_if_available = self
                        request.accept()
                        handled = True
                    else:
                        logger.debug(
                            f"insufficient element_quota {element_quota} for "
                            + f"{elements}. Calling deny on request: {request.id}"
                        )
                        request.destination_node_if_available = self
                        request.deny()
                        handled = True
                else:
                    logger.debug(
                        f"insufficient element_quota {element_quota} for {elements}."
                        + f"Calling deny on request: {request.id}"
                    )
                    request.destination_node_if_available = self
                    request.deny()
                    handled = True
            elif action == "deny":
                logger.debug(f"Calling deny on request: {request.id}")
                request.destination_node_if_available = self
                request.deny()
                handled = True

        # print or log rules can execute multiple times
        if "print_local" in handler or "log_local" in handler:
            # get a copy of the item

            try:
                if obj is None:
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
                    obj = response.obj
            except Exception as e:
                logger.critical(f"Exception getting object. {e}")

            log = f"> Request {request.name}:"
            if len(request.request_description) > 0:
                log += f" {request.request_description}"
            log += f"\nValue: {obj}"

            # if these are enabled output them
            if "print_local" in handler:
                print_local = handler["print_local"]
                if print_local:
                    print(log)
            if "log_local" in handler:
                log_local = handler["log_local"]
                if log_local:
                    logger.info(log)

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
                if "timeout_secs" in handler and handler["timeout_secs"] != -1:
                    if now - handler["created_time"] > handler["timeout_secs"]:
                        continue
                alive_handlers.append(handler)
        self.request_handlers = alive_handlers

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
            del self.handled_requests[req]

        alive_requests: List[RequestMessage] = []
        for request in self.requests:
            if request.timeout_secs is not None and request.timeout_secs > -1:
                if request.arrival_time is None:
                    logger.critical(f"Request has no arrival time. {request.id}")
                    request.set_arrival_time(arrival_time=time.time())
                arrival_time = getattr(request, "arrival_time", float(now))
                if now - arrival_time > request.timeout_secs:
                    # this request has expired
                    continue
            alive_requests.append(request)

        self.requests = alive_requests

    @syft_decorator(typechecking=True)
    async def run_handlers(self) -> None:
        while True:
            await asyncio.sleep(0.2)
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
