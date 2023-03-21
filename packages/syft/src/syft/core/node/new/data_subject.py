# stdlib
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple

# third party
from typing_extensions import Self

# relative
from .document_store import PartitionKey
from .response import SyftError
from .serializable import serializable
from .syft_object import SYFT_OBJECT_VERSION_1
from .syft_object import SyftObject
from .transforms import TransformContext
from .transforms import add_node_uid_for_key
from .transforms import generate_id
from .transforms import transform
from .uid import UID

NamePartitionKey = PartitionKey(key="name", type_=str)


@serializable(recursive_serde=True)
class DataSubject(SyftObject):
    # version
    __canonical_name__ = "DataSubject"
    __version__ = SYFT_OBJECT_VERSION_1

    node_uid: UID
    name: str
    description: Optional[str]
    aliases: List[str] = []

    @property
    def members(self) -> List:
        # relative
        from .api import APIRegistry

        api = APIRegistry.api_for(self.node_uid)
        if api is None:
            return SyftError(message=f"You must login to {self.node_uid}")
        members = api.services.data_subject.members_for(self.name)
        return members

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

    def _create_member_relationship(self, data_subject, _relationship_set):
        for member in data_subject.members.values():
            _relationship_set.add((data_subject, member))
            self._create_member_relationship(member, _relationship_set)

    def add_member(self, data_subject: Self) -> None:
        self.members[data_subject.name] = data_subject

    @property
    def member_relationships(self) -> Set[Tuple[str, str]]:
        relationships = set()
        self._create_member_relationship(self, relationships)
        return relationships

    def __repr__(self) -> str:
        return f"<DataSubject: {self.name}>"

    def _repr_markdown_(self) -> str:
        _repr_str = f"DataSubject: {self.name}\n"
        _repr_str += f"Description: {self.description}\n"
        _repr_str += f"Aliases: {self.aliases}\n"
        _repr_str += f"Members: {len(self.members)}\n"
        return "```python\n" + _repr_str + "\n```"


def remove_members_list(context: TransformContext) -> TransformContext:
    context.output.pop("members", [])
    return context


@transform(DataSubjectCreate, DataSubject)
def create_data_subject_to_data_subject():
    return [generate_id, remove_members_list, add_node_uid_for_key("node_uid")]
