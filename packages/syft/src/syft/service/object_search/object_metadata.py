# stdlib
from typing import List
from typing import Optional

# third party
from result import Result

# relative
from ...node.credentials import SyftVerifyKey
from ...serde import serializable
from ...store.document_store import BaseStash
from ...store.document_store import DocumentStore
from ...store.document_store import PartitionKey
from ...store.document_store import PartitionSettings
from ...types.syft_object import PartialSyftObject
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftBaseObject
from ..action.action_permissions import ActionObjectPermission
from ..user.user import User


@serializable()
class SyftObjectMetadata(PartialSyftObject):
    __canonical_name__ = "SyftObjectMetadata"
    __version__ = SYFT_OBJECT_VERSION_1

    canonical_name: str
    klass_version: int
    object_hash: str

    @classmethod
    def from_klass(cls, klass: SyftBaseObject):
        object_hash = cls.__generate_hash_(klass=klass)
        return SyftObjectMetadata(
            canonical_name=klass.__canonical_name__,
            klass_version=klass.__version__,
            object_hash=object_hash,
        )

    @staticmethod
    def __generate_hash_(klass: SyftBaseObject) -> str:
        unique_attrs = getattr(klass, "__attr_unique__", ())
        searchable_attrs = getattr(klass, "__attr_searchable__", ())
        return hash(
            tuple(
                klass.__fields__.values(),
                tuple(unique_attrs),
                tuple(searchable_attrs),
            )
        )


KlassNamePartitionKey = PartitionKey(key="canonical_name", type_=str)


class SyftObjectMetadataStash(BaseStash):
    object_type = SyftObjectMetadata
    settings: PartitionSettings = PartitionSettings(
        name=User.__canonical_name__,
        object_type=SyftObjectMetadata,
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    def set(
        self,
        credentials: SyftVerifyKey,
        syft_object_metadata: SyftObjectMetadata,
        add_permissions: Optional[List[ActionObjectPermission]] = None,
    ) -> Result[SyftObjectMetadata, str]:
        res = self.check_type(syft_object_metadata, self.object_type)
        # we dont use and_then logic here as it is hard because of the order of the arguments
        if res.is_err():
            return res
        return super().set(
            credentials=credentials, obj=res.ok(), add_permissions=add_permissions
        )

    def get_by_name(
        self, canonical_name: str, credentials: SyftVerifyKey
    ) -> Result[SyftObjectMetadata, str]:
        qks = KlassNamePartitionKey.with_obj(canonical_name)
        return self.query_one(credentials=credentials, qks=qks)
