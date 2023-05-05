# stdlib
import os
from typing import Optional
from typing import Type

# third party
from result import Err
from result import Ok
from result import Result
from sympy import Union

# relative
from ...client.client import SyftClient
from ...serde.serializable import serializable
from ...store.document_store import DocumentStore
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ...types.uid import UID
from ...util.telemetry import instrument
from ..context import AuthedServiceContext
from ..network.network_service import NetworkStash
from ..response import SyftError
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import service_method
from ..user.user_roles import GUEST_ROLE_LEVEL
from .headscale_client import BaseVPNClient
from .headscale_client import HeadScaleAuthToken
from .headscale_client import HeadScaleClient
from .headscale_client import VPNClientConnection
from .tailscale_client import TailScaleClient
from .tailscale_client import TailscaleStatus
from .vpn_stash import VPNStash


class NodePeerVPNKey(SyftObject):
    __canonical_name__ = "NodePeerVPNKey"
    __version__ = SYFT_OBJECT_VERSION_1

    id: Optional[UID]
    auth_key: str
    name: str


def get_vpn_client(
    client_type: Type[BaseVPNClient],
) -> Result[BaseVPNClient, str]:
    api_key = os.getenv("STACK_API_KEY")

    url = None

    if isinstance(client_type, HeadScaleClient):
        url = os.getenv("TAILSCALE_URL", "http://proxy:4000")
    elif isinstance(client_type, TailScaleClient):
        url = os.getenv("HEADSCALE_URL", "http://headscale:4000")

    if api_key and url:
        client = client_type(
            connection=VPNClientConnection(url=url),
            api_key=api_key,
        )
        return Ok(client)

    return Err(f"Failed to create client for: {client_type.__name__}")


@instrument
@serializable()
class VPNService(AbstractService):
    store: DocumentStore
    stash: NetworkStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = VPNStash(store=store)

    @service_method(path="vpn.join", name="join")
    def join_vpn(
        self,
        context: AuthedServiceContext,
        client: SyftClient,
    ) -> Union[SyftSuccess, SyftError]:
        """Join a VPN Service"""

        auth_token = client.vpn.register()

        if isinstance(auth_token, SyftError):
            return auth_token

        result = get_vpn_client(TailScaleClient)

        if result.is_err():
            return SyftError(message=result.err())

        tailscale_client = result.ok()

        result = tailscale_client.disconnect()

        if isinstance(result, SyftError):
            return result

        result = tailscale_client.connect(
            headscale_host=client.connection.url,
            headscale_auth_token=auth_token.key,
        )

        if isinstance(result, SyftError):
            return result

        result = self.stash.add_vpn_endpoint(
            context=context,
            host_or_ip=client.connection.url,
            vpn_key=auth_token.key,
        )

        if result.is_err():
            return SyftError(message=result.err())

        return SyftSuccess(message="Successfully joined VPN !!!")

    @service_method(path="vpn.status", name="status")
    def get_status(
        self,
        context: AuthedServiceContext,
    ) -> Union[TailscaleStatus, SyftError]:
        """Join a VPN Service"""
        result = get_vpn_client(TailScaleClient)

        if result.is_err():
            return SyftError(message=result.err())

        tailscale_client = result.ok()

        return tailscale_client.status()

    def connect_vpn(
        self,
        context: AuthedServiceContext,
    ) -> Union[SyftSuccess, SyftError]:
        """Join a VPN Service"""
        pass

    @service_method("register", "vpn.register", roles=[GUEST_ROLE_LEVEL])
    def register(
        self,
        context: AuthedServiceContext,
    ) -> Union[HeadScaleAuthToken, SyftError]:
        """Register node to the VPN."""

        result = get_vpn_client(HeadScaleClient)

        if result.is_err():
            return SyftError(message=result.err())

        headscale_client = result.ok()

        token = headscale_client.generate_token()

        return token
