# stdlib
from typing import Dict
from typing import List
from typing import Optional

# third party
from typing_extensions import Self

# relative
from ....core.node.common.node_table.syft_object import SYFT_OBJECT_VERSION_1
from ....core.node.common.node_table.syft_object import SyftObject
from ...common.serde.serializable import serializable
from ...common.uid import UID
from .document_store import PartitionKey
from .response import SyftError
from .transforms import TransformContext
from .transforms import generate_id
from .transforms import transform

NamePartitionKey = PartitionKey(key="name", type_=str)
# AliasesPartitionKey = PartitionKey(key="aliases", type_=List[UID])
# MembersPartitionKey = PartitionKey(key="aliases", type_=List[UID])


@serializable(recursive_serde=True)
class DataSubject(SyftObject):
    # version
    __canonical_name__ = "DataSubject"
    __version__ = SYFT_OBJECT_VERSION_1

    name: str
    description: Optional[str]
    aliases: List[str] = []
    members: Dict[str, UID] = {}

    __attr_searchable__ = ["name", "description"]
    __attr_unique__ = ["name"]

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other) -> bool:
        return hash(self) == hash(other)

    def __repr__(self) -> str:
        return f"<DataSubject: {self.name}>"

    def _repr_markdown_(self) -> str:
        _repr_str = f"DataSubject: {self.name}\n"
        _repr_str += f"Description: {self.description}\n"
        _repr_str += f"Aliases: {self.aliases}\n"
        _repr_str += f"Members: {len(self.members)}\n"
        return "```python\n" + _repr_str + "\n```"


@serializable(recursive_serde=True)
class DataSubjectCreate(SyftObject):
    # version
    __canonical_name__ = "DataSubjectCreate"
    __version__ = SYFT_OBJECT_VERSION_1

    id: Optional[UID] = None
    name: str
    description: Optional[str]
    aliases: Optional[List[str]] = []
    members: Dict[str, "DataSubjectCreate"] = {}

    __attr_searchable__ = ["name", "description"]
    __attr_unique__ = ["name"]

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other) -> bool:
        return hash(self) == hash(other)

    def add_member(self, data_subject: Self) -> None:
        self.members[data_subject.name] = data_subject

    def __repr__(self) -> str:
        return f"<DataSubject: {self.name}>"

    def _repr_markdown_(self) -> str:
        _repr_str = f"DataSubject: {self.name}\n"
        _repr_str += f"Description: {self.description}\n"
        _repr_str += f"Aliases: {self.aliases}\n"
        _repr_str += f"Members: {len(self.members)}\n"
        return "```python\n" + _repr_str + "\n```"


def members_to_ids(context: TransformContext) -> TransformContext:
    # relative
    from .data_subject_service import DataSubjectService

    members = context.output["members"]
    method = context.node.get_service_method(DataSubjectService.get_by_name)
    member_name_id_map = {}
    for name, _ in members.items():
        result = method(context=context, name=name)
        if isinstance(result, SyftError):
            return result
        ds_member = result
        member_name_id_map[name] = ds_member.id
    context.output["members"] = member_name_id_map
    return context


@transform(DataSubjectCreate, DataSubject)
def create_data_subject_to_data_subject():
    return [generate_id, members_to_ids]
