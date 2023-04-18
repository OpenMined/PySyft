# relative
from ...serde.serializable import serializable
from ...store.document_store import PartitionKey
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject

ParentPartitionKey = PartitionKey(key="parent", type_=str)
ChildPartitionKey = PartitionKey(key="child", type_=str)


@serializable()
class DataSubjectMemberRelationship(SyftObject):
    __canonical_name__ = "DataSubjectMemberRelationship"
    __version__ = SYFT_OBJECT_VERSION_1

    parent: str
    child: str

    __attr_searchable__ = ["parent", "child"]
    __attr_unique__ = ["parent", "child"]

    def __hash__(self) -> int:
        return hash(self.parent + self.child)

    def __eq__(self, other) -> bool:
        return hash(self) == hash(other)

    def __repr__(self) -> str:
        return f"<DataSubjectMembership: {self.parent} -> {self.child}>"
