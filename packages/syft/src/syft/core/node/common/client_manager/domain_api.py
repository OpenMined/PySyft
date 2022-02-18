# stdlib
import sys
import time
from typing import Any
from typing import List
from typing import Optional
from typing import Union

# third party
from pandas import DataFrame

# relative
from .....core.common.uid import UID
from .....grid.client.proxy_client import ProxyClient
from .....lib.python import String
from .....logger import error
from ....node.common import AbstractNodeClient
from ..node_service.peer_discovery.peer_discovery_messages import (
    GetPeerInfoMessageWithReply,
)
from ..node_service.peer_discovery.peer_discovery_messages import (
    PeerDiscoveryMessageWithReply,
)
from .request_api import RequestAPI


class DomainRequestAPI(RequestAPI):
    def __init__(self, client: AbstractNodeClient):
        super().__init__(client=client)
        self.cache_time = 0.0
        self.cache: Optional[List[Any]] = None
        self.num_known_domains_even_offline_ones = 0
        self.timeout = 6000  # check for new domains every 100 minutes

    def all(self, pandas: bool = True) -> List[Any]:
        response = self.perform_api_request_generic(
            syft_msg=PeerDiscoveryMessageWithReply, content={}
        )
        result = response.payload.kwargs  # type: ignore

        if result["status"] == "ok":
            _data = result["data"]
            if (
                self.cache is None
                or (time.time() - self.cache_time > self.timeout)
                or len(_data) != self.num_known_domains_even_offline_ones
            ):
                # check for logged in domains if the number of possible domains changes (if a new domain shows up)
                self.num_known_domains_even_offline_ones = len(_data)
                data = list()
                for i, domain_metadata in enumerate(_data):
                    sys.stdout.write(
                        "\rChecking whether domains are online: "
                        + str(i + 1)
                        + " of "
                        + str(len(_data))
                    )
                    try:
                        # syft absolute
                        import syft

                        syft.logger.stop()
                        if self.get(domain_metadata["id"]).ping:
                            data.append(domain_metadata)
                        syft.logger.start()
                    except Exception:  # nosec
                        # if pinging the domain causes an exception we just wont
                        # include it in the array
                        pass
                sys.stdout.write("\r                                             ")

                self.cache = data
                self.cache_time = time.time()
            else:
                data = self.cache

            if pandas:
                data = DataFrame(data)

            return data
        return []

    def get(self, key: Union[str, int, UID, String]) -> ProxyClient:  # type: ignore
        # to make sure we target the remote Domain through the proxy we need to
        # construct an 💠 Address which includes the correct UID for the Domain
        # position in the 4 hierarchical locations
        node_uid = key
        try:
            if isinstance(node_uid, int):
                domain_metadata = self.all(pandas=False)[node_uid]
                node_uid = str(domain_metadata["id"])
            elif isinstance(node_uid, String):
                node_uid = node_uid.upcast()
        except Exception as e:
            error(f"Invalid int or String key for list of Domain Clients. {e}")

        if isinstance(node_uid, UID):
            node_uid = node_uid.no_dash

        if not isinstance(node_uid, str):
            msg = (
                f"Unable to get ProxyClient with key with type {type(node_uid)} {node_uid}. "
                "API Request requires key to resolve to a str."
            )
            error(msg)
            raise Exception(msg)
        response = self.perform_api_request_generic(
            syft_msg=GetPeerInfoMessageWithReply, content={"uid": node_uid}
        )

        result = response.payload.kwargs.upcast()  # type: ignore

        # a ProxyClient requires an existing NetworkClient and a Remote Domain Address
        # or a known Domain Node UID, and a Node Name
        proxy_client = ProxyClient.create(
            proxy_node_client=self.client,
            remote_domain=node_uid,
            domain_name=result["data"]["name"],
        )

        return proxy_client

    def __getitem__(self, key: Union[str, int, UID]) -> ProxyClient:
        return self.get(key=key)
