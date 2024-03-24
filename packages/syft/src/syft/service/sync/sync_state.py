# stdlib
import html
from typing import Any
from typing import Optional
from typing import TYPE_CHECKING

# relative
from ...serde.serializable import serializable
from ...store.linked_obj import LinkedObject
from ...types.datetime import DateTime
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ...types.syncable_object import SyncableSyftObject
from ...types.uid import LineageID
from ...types.uid import UID

if TYPE_CHECKING:
    # relative
    from .diff_state import NodeDiff


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
    previous_object: SyftObject | None = None
    current_state: str
    previous_state: str
    level: int = 0

    # TODO table formatting
    __repr_attrs__ = [
        "previous_state",
        "current_state",
    ]

    def _coll_repr_(self) -> dict[str, Any]:
        current_state = f"{self.status}\n{self.current_state}"
        previous_state = f"{self.status}\n{self.previous_state}"
        return {
            "previous_state": html.escape(previous_state),
            "current_state": html.escape(current_state),
        }

    @property
    def object_type(self) -> str:
        prefix = get_hierarchy_level_prefix(self.level)
        return f"{prefix}{type(self.object).__name__}"

    @property
    def status(self) -> str:
        # TODO use Diffs to determine status
        if self.previous_object is None:
            return "NEW"
        elif self.previous_object.syft_eq(ext_obj=self.object):
            return "SAME"
        else:
            return "UPDATED"


@serializable()
class SyncState(SyftObject):
    __canonical_name__ = "SyncState"
    __version__ = SYFT_OBJECT_VERSION_1

    node_uid: UID
    objects: dict[UID, SyncableSyftObject] = {}
    dependencies: dict[UID, list[UID]] = {}
    created_at: DateTime = DateTime.now()
    previous_state_link: LinkedObject | None = None
    permissions: dict[UID, set[str]] = {}
    storage_permissions: dict[UID, set[UID]] = {}

    __attr_searchable__ = ["created_at"]

    @property
    def previous_state(self) -> Optional["SyncState"]:
        if self.previous_state_link is not None:
            return self.previous_state_link.resolve
        return None

    @property
    def all_ids(self) -> set[UID]:
        return set(self.objects.keys())

    def add_objects(self, objects: list[SyncableSyftObject], api: Any = None) -> None:
        for obj in objects:
            if isinstance(obj.id, LineageID):
                self.objects[obj.id.id] = obj
            else:
                self.objects[obj.id] = obj

        # TODO might get slow with large states,
        # need to build dependencies every time to not have UIDs
        # in dependencies that are not in objects
        self._build_dependencies(api=api)

    def _build_dependencies(self, api: Any = None) -> None:
        self.dependencies = {}

        all_ids = self.all_ids
        for obj in self.objects.values():
            if hasattr(obj, "get_sync_dependencies"):
                deps = obj.get_sync_dependencies(api=api)
                deps = [d.id for d in deps if d.id in all_ids]  # type: ignore
                if len(deps):
                    self.dependencies[obj.id] = deps

    def get_previous_state_diff(self) -> "NodeDiff":
        # Re-use DiffState to compare to previous state
        # Low = previous, high = current
        # relative
        from .diff_state import NodeDiff

        previous_state = self.previous_state or SyncState(node_uid=self.node_uid)
        return NodeDiff.from_sync_state(previous_state, self)

    @property
    def rows(self) -> list[SyncStateRow]:
        result = []
        ids = set()

        previous_diff = self.get_previous_state_diff()
        for hierarchy in previous_diff.hierarchies:
            for diff, level in zip(hierarchy.diffs, hierarchy.hierarchy_levels):
                if diff.object_id in ids:
                    continue
                ids.add(diff.object_id)
                row = SyncStateRow(
                    object=diff.high_obj,
                    previous_object=diff.low_obj,
                    current_state=diff.diff_side_str("high"),
                    previous_state=diff.diff_side_str("low"),
                    level=level,
                )
                result.append(row)
        return result

    def _repr_html_(self) -> str:
        return self.rows._repr_html_()
