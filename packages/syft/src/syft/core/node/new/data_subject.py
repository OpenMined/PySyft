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
from .document_store import PartitionKey

NamePartitionKey = PartitionKey(key="name", type_=str)


@serializable(recursive_serde=True)
class DataSubject(SyftObject):
    # version
    __canonical_name__ = "DataSubject"
    __version__ = SYFT_OBJECT_VERSION_1

    name: str
    description: Optional[str]
    aliases: Optional[List[str]]
    members: Dict[str, "DataSubject"] = {}

    __attr_searchable__ = ["name", "aliases", "description"]
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


@serializable(recursive_serde=True)
class DataSubjectRegistry(SyftObject):
    # version
    __canonical_name__ = "DataSubjectRegistry"
    __version__ = SYFT_OBJECT_VERSION_1

    data_subjects: List[DataSubject] = []

    def add_data_subject(self, data_subject: DataSubject) -> None:
        # TODO: Check for uniqueness, it could happen member is already
        # registered, therefore, the registry could be a set instead of
        # list -> Need to confirm the representation
        self.data_subjects.append(data_subject)
        for _, member in data_subject.members.items():
            self.add_data_subject(member)

    def _repr_markdown_(self) -> str:
        _repr_str = f"DataSubjects: {len(self.data_subjects)}\n"
        return "```python\n" + _repr_str + "\n```"
