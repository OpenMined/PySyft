# stdlib
from typing import Optional

# third party
from sympy import Union

# relative
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
from .vpn_stash import VPNStash


class NodePeerVPNKey(SyftObject):
    __canonical_name__ = "NodePeerVPNKey"
    __version__ = SYFT_OBJECT_VERSION_1

    id: Optional[UID]
    auth_key: str
    name: str


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
    ) -> Union[SyftSuccess, SyftError]:
        """Join a VPN Service"""
        # node_id = node.node.create_or_get_node(  # type: ignore
        #     node_uid=res_json["node_id"],
        #     node_name=res_json["node_name"],
        # )

        # self.stash.add_vpn_endpoint(  # type: ignore
        #     context=context
        #     host_or_ip=res_json["host_or_ip"],
        #     vpn_endpoint=str(grid_url.with_path("/vpn")),
        #     vpn_key=res_json["vpn_auth_key"],
        # )
        pass

    @service_method(path="vpn.status", name="status")
    def get_status(
        self,
        context: AuthedServiceContext,
    ) -> Union[SyftSuccess, SyftError]:
        """Join a VPN Service"""
        pass

    def connect_vpn(
        self,
        context: AuthedServiceContext,
    ) -> Union[SyftSuccess, SyftError]:
        """Join a VPN Service"""
        pass

    def register(
        self,
        context: AuthedServiceContext,
    ) -> Union[SyftSuccess, SyftError]:
        """Register node to the VPN."""
        pass
