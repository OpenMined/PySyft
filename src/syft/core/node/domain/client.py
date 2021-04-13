# stdlib
import time
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

# third party
from nacl.signing import SigningKey
from nacl.signing import VerifyKey
import pandas as pd

# syft relative
from ....logger import traceback_and_raise
from ....util import validate_field
from ...common.uid import UID
from ...io.location import Location
from ...io.location import SpecificLocation
from ...io.route import Route
from ..common.client import Client
from .service import RequestMessage


class RequestQueueClient:
    def __init__(self, client: Client) -> None:
        self.client = client
        self.handlers = RequestHandlerQueueClient(client=client)

    @property
    def requests(self) -> List[RequestMessage]:
        # syft absolute
        from syft.core.node.domain.service.get_all_requests_service import (
            GetAllRequestsMessage,
        )

        msg = GetAllRequestsMessage(
            address=self.client.address, reply_to=self.client.address
        )

        requests = self.client.send_immediate_msg_with_reply(msg=msg).requests  # type: ignore

        for request in requests:
            request.gc_enabled = False
            request.owner_client_if_available = self.client

        return requests

    def get_request_id_from_object_id(self, object_id: UID) -> Optional[UID]:
        for req in self.requests:
            if req.object_id == object_id:
                return req.request_id

        return object_id

    def __getitem__(self, key: Union[str, int]) -> RequestMessage:
        if isinstance(key, str):
            for request in self.requests:
                if key == str(request.id.value):
                    return request
            traceback_and_raise(
                KeyError("No such request found for string id:" + str(key))
            )
        if isinstance(key, int):
            return self.requests[key]
        else:
            traceback_and_raise(KeyError("Please pass in a string or int key"))

        raise Exception("should not get here")

    def __repr__(self) -> str:
        return repr(self.requests)

    @property
    def pandas(self) -> pd.DataFrame:
        request_lines = [
            {
                "Requested Object's tags": request.object_tags,
                "Reason": request.request_description,
                "Request ID": request.id,
                "Requested Object's ID": request.object_id,
                "Requested Object's type": request.object_type,
            }
            for request in self.requests
        ]
        return pd.DataFrame(request_lines)

    def add_handler(
        self,
        action: str,
        print_local: bool = False,
        log_local: bool = False,
        tags: Optional[List[str]] = None,
        timeout_secs: int = -1,
        element_quota: Optional[int] = None,
    ) -> None:
        handler_opts = self._validate_options(
            id=UID(),
            action=action,
            print_local=print_local,
            log_local=log_local,
            tags=tags,
            timeout_secs=timeout_secs,
            element_quota=element_quota,
        )

        self._update_handler(handler_opts, keep=True)

    def remove_handler(self, key: Union[str, int]) -> None:
        handler_opts = self.handlers[key]

        self._update_handler(handler_opts, keep=False)

    def clear_handlers(self) -> None:
        for handler in self.handlers.handlers:
            id_str = str(handler["id"].value).replace("-", "")
            self.remove_handler(id_str)

    def _validate_options(
        self,
        action: str,
        print_local: bool = False,
        log_local: bool = False,
        tags: Optional[List[str]] = None,
        timeout_secs: int = -1,
        element_quota: Optional[int] = None,
        id: Optional[UID] = None,
    ) -> Dict[str, Any]:
        handler_opts: Dict[str, Any] = {}
        if action not in ["accept", "deny"]:
            traceback_and_raise(Exception("Action must be 'accept' or 'deny'"))
        handler_opts["action"] = action
        handler_opts["print_local"] = bool(print_local)
        handler_opts["log_local"] = bool(log_local)

        handler_opts["tags"] = []
        if tags is not None:
            for tag in tags:
                handler_opts["tags"].append(tag)
        handler_opts["timeout_secs"] = max(-1, int(timeout_secs))
        if element_quota is not None:
            handler_opts["element_quota"] = max(0, int(element_quota))

        if id is None:
            id = UID()
        handler_opts["id"] = id

        return handler_opts

    def _update_handler(self, request_handler: Dict[str, Any], keep: bool) -> None:
        # syft absolute
        from syft.core.node.domain.service.request_handler_service import (
            UpdateRequestHandlerMessage,
        )

        msg = UpdateRequestHandlerMessage(
            address=self.client.address, handler=request_handler, keep=keep
        )
        self.client.send_immediate_msg_without_reply(msg=msg)


class RequestHandlerQueueClient:
    def __init__(self, client: Client) -> None:
        self.client = client

    @property
    def handlers(self) -> List[Dict]:
        # syft absolute
        from syft.core.node.domain.service.request_handler_service import (
            GetAllRequestHandlersMessage,
        )

        msg = GetAllRequestHandlersMessage(
            address=self.client.address, reply_to=self.client.address
        )
        return validate_field(
            self.client.send_immediate_msg_with_reply(msg=msg), "handlers"
        )

    def __getitem__(self, key: Union[str, int]) -> Dict:
        """
        allow three ways to get an request handler:
            1. use id: str
            2. use tag: str
            3. use index: int
        """
        if isinstance(key, str):
            matches = 0
            match_handler: Optional[Dict] = None
            for handler in self.handlers:
                if key in str(handler["id"].value).replace("-", ""):
                    return handler
                if key in handler["tags"]:
                    matches += 1
                    match_handler = handler
            if matches == 1 and match_handler is not None:
                return match_handler
            elif matches > 1:
                raise KeyError("More than one item with tag:" + str(key))

            raise KeyError("No such request found for string id:" + str(key))
        if isinstance(key, int):
            return self.handlers[key]
        else:
            raise KeyError("Please pass in a string or int key")

    def __repr__(self) -> str:
        return repr(self.handlers)

    @property
    def pandas(self) -> pd.DataFrame:
        def _get_time_remaining(handler: dict) -> int:
            timeout_secs = handler.get("timeout_secs", -1)
            if timeout_secs == -1:
                return -1
            else:
                created_time = handler.get("created_time", 0)
                rem = timeout_secs - (time.time() - created_time)
                return round(rem)

        handler_lines = [
            {
                "tags": handler["tags"],
                "ID": handler["id"],
                "action": handler["action"],
                "remaining time (s):": _get_time_remaining(handler),
            }
            for handler in self.handlers
        ]
        return pd.DataFrame(handler_lines)


class DomainClient(Client):

    domain: SpecificLocation
    requests: RequestQueueClient

    def __init__(
        self,
        name: Optional[str],
        routes: List[Route],
        domain: SpecificLocation,
        network: Optional[Location] = None,
        device: Optional[Location] = None,
        vm: Optional[Location] = None,
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

        self.requests = RequestQueueClient(client=self)
        self.post_init()

    @property
    def id(self) -> UID:
        return self.domain.id

    @property
    def device(self) -> Optional[Location]:
        """This client points to a node, if that node lives within a device
        or is a device itself, this property will return the Location of that device
        if it is known by the client."""

        return super().device

    @device.setter
    def device(self, new_device: Location) -> Optional[Location]:
        """This client points to a node, if that node lives within a device
        or is a device itself and we learn the Location of that device, this setter
        allows us to save the Location of that device for use later. We use a getter
        (@property) and setter (@set) explicitly because we want all clients
        to efficiently save an address object for use when sending messages to their
        target. That address object will include this information if it is available"""

        traceback_and_raise(
            Exception("This client points to a domain, you don't need a Device ID.")
        )

    @property
    def vm(self) -> Optional[Location]:
        """This client points to a node, if that node lives within a vm
        or is a vm itself, this property will return the Location of that vm
        if it is known by the client."""

        return super().vm

    @vm.setter
    def vm(self, new_vm: Location) -> Optional[Location]:
        """This client points to a node, if that node lives within a vm
        or is a vm itself and we learn the Location of that vm, this setter
        allows us to save the Location of that vm for use later. We use a getter
        (@property) and setter (@set) explicitly because we want all clients
        to efficiently save an address object for use when sending messages to their
        target. That address object will include this information if it is available"""

        traceback_and_raise(
            Exception("This client points to a device, you don't need a VM Location.")
        )

    def __repr__(self) -> str:
        no_dash = str(self.id).replace("-", "")
        return f"<{type(self).__name__}: {no_dash}>"

    def update_vars(self, state: dict) -> pd.DataFrame:
        for ptr in self.store.store:
            tags = getattr(ptr, "tags", None)
            if tags is not None:
                for tag in tags:
                    state[tag] = ptr
        return self.store.pandas
