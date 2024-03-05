# stdlib
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple

# third party
from typing_extensions import Self

# relative
from ...serde.serializable import serializable
from ...store.document_store import PartitionKey
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ...types.transforms import TransformContext
from ...types.transforms import add_node_uid_for_key
from ...types.transforms import generate_id
from ...types.transforms import transform
from ...types.uid import UID
from ...util.markdown import as_markdown_python_code
from ..response import SyftError

NamePartitionKey = PartitionKey(key="name", type_=str)


@serializable()
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
        from ...client.api import APIRegistry

        api = APIRegistry.api_for(self.node_uid, self.syft_client_verify_key)
        if api is None:
            return SyftError(message=f"You must login to {self.node_uid}")
        members = api.services.data_subject.members_for(self.name)
        return members

    __attr_searchable__ = ["name", "description"]
    __repr_attrs__ = ["name", "description"]
    __attr_unique__ = ["name"]

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: Any) -> bool:
        return hash(self) == hash(other)

    def __repr_syft_nested__(self) -> str:
        return f"DataSubject({self.name})"

    def __repr__(self) -> str:
        return f"<DataSubject: {self.name}>"

    def _repr_markdown_(self, wrap_as_python: bool = True, indent: int = 0) -> str:
        _repr_str = f"DataSubject: {self.name}\n"
        _repr_str += f"Description: {self.description}\n"
        _repr_str += f"Aliases: {self.aliases}\n"
        _repr_str += f"Members: {len(self.members)}\n"
        return as_markdown_python_code(_repr_str)


@serializable()
class DataSubjectCreate(SyftObject):
    # version
    __canonical_name__ = "DataSubjectCreate"
    __version__ = SYFT_OBJECT_VERSION_1

    id: Optional[UID] = None  # type: ignore[assignment]
    name: str
    description: Optional[str]
    aliases: Optional[List[str]] = []
    members: Dict[str, "DataSubjectCreate"] = {}

    __attr_searchable__ = ["name", "description"]
    __attr_unique__ = ["name"]
    __repr_attrs__ = ["name", "member_count"]

    @property
    def member_count(self) -> int:
        return len(self.members)

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: Any) -> bool:
        return hash(self) == hash(other)

    def __repr_syft_nested__(self) -> str:
        return f"DataSubject({self.name})"

    def _create_member_relationship(
        self, data_subject: Self, _relationship_set: set
    ) -> None:
        for member in data_subject.members.values():
            _relationship_set.add((data_subject, member))
            self._create_member_relationship(member, _relationship_set)

    def add_member(self, data_subject: Self) -> None:
        self.members[data_subject.name] = data_subject

    @property
    def member_relationships(self) -> Set[Tuple[str, str]]:
        relationships: set = set()
        self._create_member_relationship(self, relationships)
        return relationships

    def __repr__(self) -> str:
        return f"<DataSubject: {self.name}>"

    def _repr_markdown_(self, wrap_as_python: bool = True, indent: int = 0) -> str:
        _repr_str = f"DataSubject: {self.name}\n"
        _repr_str += f"Description: {self.description}\n"
        _repr_str += f"Aliases: {self.aliases}\n"
        _repr_str += f"Members: {len(self.members)}\n"
        return as_markdown_python_code(_repr_str)


def remove_members_list(context: TransformContext) -> TransformContext:
    if context.output is not None:
        context.output.pop("members", [])
    return context


@transform(DataSubjectCreate, DataSubject)
def create_data_subject_to_data_subject() -> list[Callable]:
    return [generate_id, remove_members_list, add_node_uid_for_key("node_uid")]
