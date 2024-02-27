# stdlib
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import TYPE_CHECKING
from typing import Tuple

# relative
from ...serde.serializable import serializable
from ...store.linked_obj import LinkedObject
from ...types.datetime import DateTime
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ...types.uid import LineageID
from ...types.uid import UID

if TYPE_CHECKING:
    # relative
    from .diff_state import DiffState


def get_hierarchy_level_prefix(level: int) -> str:
    if level == 0:
        return ""
    else:
        return "--" * level + " "


@serializable()
class SyncStateRow(SyftObject):
    """A row in the SyncState table"""

    __canonical_name__ = "SyncStateItem"
    __version__ = SYFT_OBJECT_VERSION_1

    object: SyftObject
    previous_object: Optional[SyftObject]
    current_state: str
    previous_state: str
    level: int = 0

    # TODO table formatting
    __repr_attrs__ = [
        "object_type",
        "status",
        "previous_state",
        "current_state",
    ]

    @property
    def object_type(self) -> str:
        prefix = get_hierarchy_level_prefix(self.level)
        return f"{prefix}{type(self.object).__name__}"

    @property
    def status(self) -> str:
        # TODO use Diffs to determine status
        if self.previous_object is None:
            return "NEW"
        else:
            return "UPDATED"


@serializable()
class SyncState(SyftObject):
    __canonical_name__ = "SyncState"
    __version__ = SYFT_OBJECT_VERSION_1

    objects: Dict[UID, SyftObject] = {}
    dependencies: Dict[UID, List[UID]] = {}
    created_at: DateTime = DateTime.now()
    previous_state_link: Optional[LinkedObject] = None

    __attr_searchable__ = ["created_at"]

    @property
    def previous_state(self) -> Optional["SyncState"]:
        if self.previous_state_link is not None:
            return self.previous_state_link.resolve

    @property
    def all_ids(self) -> Set[UID]:
        return set(self.objects.keys())

    def add_objects(self, objects: List[SyftObject], api=None) -> None:
        for obj in objects:
            if isinstance(obj.id, LineageID):
                self.objects[obj.id.id] = obj
            else:
                self.objects[obj.id] = obj

        # TODO might get slow with large states,
        # need to build dependencies every time to not have UIDs
        # in dependencies that are not in objects
        self._build_dependencies(api=api)

    def _build_dependencies(self, api=None) -> None:
        self.dependencies = {}

        all_ids = self.all_ids
        for obj in self.objects.values():
            if hasattr(obj, "get_sync_dependencies"):
                deps = obj.get_sync_dependencies(api=api)
                deps = [d.id for d in deps if d.id in all_ids]
                if len(deps):
                    self.dependencies[obj.id] = deps

    @property
    def hierarchies(self) -> List[List[Tuple[SyftObject, int]]]:
        def _build_hierarchy_helper(uid: UID, level: int = 0) -> List[Tuple[UID, int]]:
            result = [(uid, level)]
            if uid in self.dependencies:
                for child_uid in self.dependencies[uid]:
                    result.extend(_build_hierarchy_helper(child_uid, level + 1))
            return result

        result = []
        all_ids = self.all_ids
        child_ids = {child for deps in self.dependencies.values() for child in deps}
        root_ids = list(all_ids - child_ids)

        for root_uid in root_ids:
            uid_hierarchy = _build_hierarchy_helper(root_uid)
            object_hierarchy = [
                (self.objects[uid], level) for uid, level in uid_hierarchy
            ]
            result.append(object_hierarchy)

        return result

    def get_previous_state_diff(self) -> "DiffState":
        # Re-use DiffState to compare to previous state
        # Low = previous, high = current
        # relative
        from .diff_state import DiffState

        previous_state = self.previous_state or SyncState()
        return DiffState.from_sync_state(previous_state, self)

    @property
    def rows(self) -> List[SyncStateRow]:
        result = []

        previous_diff = self.get_previous_state_diff()
        for hierarchy in previous_diff.hierarchies:
            for diff, level in hierarchy:
                row = SyncStateRow(
                    object=diff.high_obj,
                    previous_object=diff.low_obj,
                    current_state=diff.high_state,
                    previous_state=diff.low_state,
                    level=level,
                )
                result.append(row)
        return result

    def _repr_html_(self):
        return self.rows._repr_html_()
