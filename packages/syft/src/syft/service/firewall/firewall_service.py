# stdlib
from ipaddress import IPv4Address
from typing import Optional
from typing import Union

# third party
from result import Result
from typing_extensions import Self

# relative
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
from ...store.document_store import BaseUIDStoreStash
from ...store.document_store import DocumentStore
from ...store.document_store import PartitionKey
from ...store.document_store import PartitionSettings
from ...store.document_store import QueryKeys
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ...util.telemetry import instrument
from ..context import AuthedServiceContext
from ..response import SyftError
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import service_method


@serializable()
class IPAddr(SyftObject):
    __canonical_name__ = "IPAddr"
    __version__ = SYFT_OBJECT_VERSION_1

    ip: IPv4Address

    def from_str(ip: str) -> Self:
        return IPAddr(ip=IPv4Address(ip))

    def __repr__(self) -> str:
        return str(self.ip)

    def __str__(self) -> str:
        return str(self.ip)

    def __hash__(self) -> int:
        return hash(str(self))


IpAddrPartitionPartitionKey = PartitionKey(key="ipaddr", type_=IPAddr)


@serializable()
class FirewallRule(SyftObject):
    __canonical_name__ = "FirewallRule"
    __version__ = SYFT_OBJECT_VERSION_1

    ipaddr: IPAddr
    blocked: bool = True

    __attr_searchable__ = ["ipaddr"]
    __attr_unique__ = ["ipaddr"]
    __attr_repr_cols__ = ["ipaddr", "blocked"]


@instrument
@serializable()
class FirewallStash(BaseUIDStoreStash):
    object_type = FirewallRule
    settings: PartitionSettings = PartitionSettings(
        name=FirewallRule.__canonical_name__, object_type=FirewallRule
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    def get_by_ip(
        self, credentials: SyftVerifyKey, ip_addr: IPAddr
    ) -> Result[Optional[FirewallRule], SyftError]:
        qks = QueryKeys(qks=[IpAddrPartitionPartitionKey.with_obj(ip_addr)])
        return self.query_one(credentials=credentials, qks=qks)


@instrument
@serializable()
class FirewallService(AbstractService):
    store: DocumentStore
    stash: FirewallStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = FirewallStash(store=store)

    @service_method(path="firewall.toggle_ip_firewall", name="toggle_ip_firewall")
    def get_by_ip(
        self,
        context: AuthedServiceContext,
        ip: str,
    ) -> Union[SyftSuccess, SyftError]:
        ip_addr = IPAddr.from_str(ip)
        result = self.stash.get_by_ip(context.credentials, ip_addr)
        if result.is_err():
            return result
        return result
