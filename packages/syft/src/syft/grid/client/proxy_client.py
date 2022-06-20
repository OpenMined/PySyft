# future
from __future__ import annotations

# stdlib
from typing import Union

# third party
from nacl.encoding import HexEncoder
from nacl.signing import SigningKey

# relative
from ...core.common.uid import UID
from ...core.io.address import Address
from ...core.io.location import SpecificLocation
from ...core.io.route import SoloRoute
from ...core.node.common import AbstractNodeClient
from ...core.node.common.node_service.ping.ping_messages import PingMessageWithReply
from ...core.node.domain_client import DomainClient
from ...logger import error
from .grid_connection import GridHTTPConnection


class ProxyClient(DomainClient):
    def post_init(self) -> None:
        # use post_init so we don't need to re-implement an empty regular init
        super().post_init()
        self.logged_in = False

    def __repr__(self) -> str:
        if self.logged_in:
            return super().__repr__()
        else:
            return (
                f"(This is a logged out ProxyClient() object for a domain called '{self.name}'."
                f" Please call .login(email, password) to get a full client you can use for stuff.)"
            )

    @property
    def ping(self) -> bool:
        try:
            # Build Syft Message
            msg = (
                PingMessageWithReply(kwargs={"host_or_ip": "asdf"})
                .to(address=self.address, reply_to=self.address)
                .sign(signing_key=self.signing_key)
            )
            self.send_immediate_msg_with_reply(msg, timeout=1)
            return True
        except Exception:
            return False

    @staticmethod
    def create(
        proxy_node_client: AbstractNodeClient,
        remote_domain: Union[str, UID, Address],
        domain_name: str,
    ) -> ProxyClient:
        domain_address = remote_domain
        try:
            if isinstance(domain_address, str):
                domain_address = UID.from_string(value=domain_address)
        except Exception as e:
            error(f"Failed to convert remote_domain str to UID. {e}")

        try:
            if isinstance(domain_address, UID):
                spec_location = SpecificLocation(domain_address)
                domain_address = Address(name=domain_name, domain=spec_location)
        except Exception as e:
            error(f"Failed to convert remote_domain UID to Address. {e}")

        if not isinstance(domain_address, Address):
            raise Exception(
                f"Failed to convert remote_domain {domain_address} to Address"
            )

        # the DomainRequestAPI is only on Network Clients so here we can re-use
        # the existing client route which will be pointing to the current Network
        base_url = proxy_node_client.routes[0].connection.base_url  # type: ignore
        conn = GridHTTPConnection(url=base_url)

        # this will initially be a guest connection so we should generate a new
        # unique key
        _user_key = SigningKey.generate()

        # Create a new Solo Route using the selected connection type
        route = SoloRoute(destination=spec_location, connection=conn)

        # here we mix all of the above together into a ProxyClient which is like a
        # normal client, except the route and the address are for different nodes
        proxy_client = ProxyClient(
            name=domain_name,
            routes=[route],
            signing_key=_user_key,
            domain=domain_address.domain,  # type: ignore
        )

        return proxy_client

    def login(self, email: str, password: str) -> ProxyClient:
        result = self.users.login(email=email, password=password)
        try:
            if result["status"] == "ok":
                _user_key = SigningKey(
                    result["data"]["key"].encode(), encoder=HexEncoder
                )
                self.signing_key = _user_key
                self.verify_key = self.signing_key.verify_key
                print(f"Logged in to {self.name} as {email}")
                self.logged_in = True
            else:
                print(f"Failed to login as {email}")
        except Exception as e:
            print(f"Failed to decode private key. {e}")
        return self

    def logout(self) -> ProxyClient:
        print(f"Logged out of account. You are now a Guest on {self.name}")
        self.signing_key = SigningKey.generate()
        self.verify_key = self.signing_key.verify_key
        self.logged_in = False
        return self
