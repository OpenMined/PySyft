# third party
from sympy import Union

# relative
from ...serde.serializable import serializable
from ...store.document_store import DocumentStore
from ...util.telemetry import instrument
from ..context import AuthedServiceContext
from ..response import SyftError
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import service_method
from .vpn_stash import VPNStash


@instrument
@serializable()
class VPNService(AbstractService):
    store: DocumentStore
    stash: VPNStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = VPNStash(store=store)

    @service_method(path="vpn.join", name="join")
    def join_vpn(
        self,
        context: AuthedServiceContext,
    ) -> Union[SyftSuccess, SyftError]:
        """Join a VPN Service"""
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
        """Join a VPN Service"""
        pass
