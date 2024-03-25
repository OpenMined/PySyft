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
from ..response import SyftError

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
    previous_state_diff: "NodeDiff" | None = None  # type: ignore

    __attr_searchable__ = ["created_at"]

    @classmethod
    def from_objects(
        cls,
        node_uid: UID,
        objects: list[SyncableSyftObject],
        permissions: dict[UID, set[str]],
        storage_permissions: dict[UID, set[UID]],
        previous_state_link: LinkedObject | None = None,
    ) -> "SyncState":
        state = cls(
            node_uid=node_uid,
            previous_state_link=previous_state_link,
        )

        state._add_objects(objects)
        return state

    def _set_previous_state_diff(self) -> None:
        # relative
        from .diff_state import NodeDiff

        # Re-use NodeDiff to compare to previous state
        # Low = previous state, high = current state
        # NOTE No previous sync state means everything is new
        previous_state = self.previous_state or SyncState(node_uid=self.node_uid)
        self.previous_state_diff = NodeDiff.from_sync_state(previous_state, self)

    @property
    def previous_state(self) -> Optional["SyncState"]:
        if self.previous_state_link is not None:
            return self.previous_state_link.resolve
        return None

    @property
    def all_ids(self) -> set[UID]:
        return set(self.objects.keys())

    def get_status(self, uid: UID) -> str | None:
        if self.previous_state_diff is None:
            return None
        diff = self.previous_state_diff.obj_uid_to_diff.get(uid)

        if diff is None:
            return None
        return diff.status

    def _add_objects(self, objects: list[SyncableSyftObject]) -> None:
        for obj in objects:
            if isinstance(obj.id, LineageID):
                self.objects[obj.id.id] = obj
            else:
                self.objects[obj.id] = obj

        # TODO might get slow with large states,
        # need to build dependencies every time to not have UIDs
        # in dependencies that are not in objects
        self._build_dependencies()
        self._set_previous_state_diff()

    def _build_dependencies(self) -> None:
        self.dependencies = {}

        all_ids = self.all_ids
        for obj in self.objects.values():
            if hasattr(obj, "get_sync_dependencies"):
                deps = obj.get_sync_dependencies()
                if isinstance(deps, SyftError):
                    return deps

                deps = [d.id for d in deps if d.id in all_ids]  # type: ignore

                if len(deps):
                    self.dependencies[obj.id] = deps

    @property
    def rows(self) -> list[SyncStateRow]:
        result = []
        ids = set()

        previous_diff = self.previous_state_diff
        if previous_diff is None:
            raise ValueError("No previous state to compare to")
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
