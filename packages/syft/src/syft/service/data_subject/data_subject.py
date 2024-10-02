# stdlib
from collections.abc import Callable
from typing import Any

# third party
from typing_extensions import Self

# relative
from ...serde.serializable import serializable
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ...types.transforms import TransformContext
from ...types.transforms import add_server_uid_for_key
from ...types.transforms import generate_id
from ...types.transforms import transform
from ...types.uid import UID
from ...util.markdown import as_markdown_python_code


@serializable()
class DataSubject(SyftObject):
    # version
    __canonical_name__ = "DataSubject"
    __version__ = SYFT_OBJECT_VERSION_1

    server_uid: UID
    name: str
    description: str | None = None
    aliases: list[str] = []

    @property
    def members(self) -> list:
        # relative
        return self.get_api().services.data_subject.members_for(self.name)

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

    id: UID | None = None  # type: ignore[assignment]
    name: str
    description: str | None = None
    aliases: list[str] | None = []
    members: dict[str, "DataSubjectCreate"] = {}

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
    def member_relationships(self) -> set[tuple[str, str]]:
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
    return [generate_id, remove_members_list, add_server_uid_for_key("server_uid")]
