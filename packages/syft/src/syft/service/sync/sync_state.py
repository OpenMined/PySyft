# stdlib
from datetime import timedelta
from typing import Any
from typing import Optional

# third party
from pydantic import Field

# relative
from ...abstract_server import ServerSideType
from ...serde.serializable import serializable
from ...store.linked_obj import LinkedObject
from ...types.datetime import DateTime
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ...types.syncable_object import SyncableSyftObject
from ...types.uid import LineageID
from ...types.uid import UID
from ...util.notebook_ui.components.sync import SyncTableObject
from ..context import AuthedServiceContext


def get_hierarchy_level_prefix(level: int) -> str:
    if level == 0:
        return ""
    else:
        return "--" * level + " "


class SyncStateRow(SyftObject):
    """A row in the SyncState table"""

    __canonical_name__ = "SyncStateItem"
    __version__ = SYFT_OBJECT_VERSION_1

    object: SyftObject
    previous_object: SyftObject | None = None
    current_state: str
    previous_state: str
    status: str
    level: int = 0
    last_sync_date: DateTime | None = None

    __syft_include_id_coll_repr__ = False

    # TODO table formatting
    __repr_attrs__ = [
        "previous_state",
        "current_state",
    ]
    __table_coll_widths__ = ["min-content", "auto", "auto", "auto"]

    def status_badge(self) -> dict[str, str]:
        status = self.status
        if status == "NEW":
            badge_color = "label-green"
        elif status == "SAME":
            badge_color = "label-gray"
        else:
            badge_color = "label-orange"
        return {"value": status.upper(), "type": badge_color}

    def _coll_repr_(self) -> dict[str, Any]:
        obj_view = SyncTableObject(object=self.object)

        if self.last_sync_date is not None:
            last_sync_date = self.last_sync_date
            last_sync_delta = timedelta(
                seconds=DateTime.now().utc_timestamp - last_sync_date.utc_timestamp
            )
            last_sync_delta_str = td_format(last_sync_delta)
            last_sync_html = (
                f"<p class='diff-state-no-obj'>{last_sync_delta_str} ago</p>"
            )
        else:
            last_sync_html = "<p class='diff-state-no-obj'>n/a</p>"
        return {
            "Status": self.status_badge(),
            "Summary": obj_view.to_html(),
            "Last Sync": last_sync_html,
        }

    @property
    def object_type(self) -> str:
        prefix = get_hierarchy_level_prefix(self.level)
        return f"{prefix}{type(self.object).__name__}"


def td_format(td_object: timedelta) -> str:
    seconds = int(td_object.total_seconds())
    if seconds == 0:
        return "0 seconds"

    periods = [
        ("year", 60 * 60 * 24 * 365),
        ("month", 60 * 60 * 24 * 30),
        ("day", 60 * 60 * 24),
        ("hour", 60 * 60),
        ("minute", 60),
        ("second", 1),
    ]

    strings = []
    for period_name, period_seconds in periods:
        if seconds >= period_seconds:
            period_value, seconds = divmod(seconds, period_seconds)
            has_s = "s" if period_value > 1 else ""
            strings.append(f"{period_value} {period_name}{has_s}")

    return ", ".join(strings)


@serializable()
class SyncState(SyftObject):
    __canonical_name__ = "SyncState"
    __version__ = SYFT_OBJECT_VERSION_1

    server_uid: UID
    server_name: str
    server_side_type: ServerSideType
    objects: dict[UID, SyncableSyftObject] = {}
    dependencies: dict[UID, list[UID]] = {}
    created_at: DateTime = Field(default_factory=DateTime.now)
    previous_state_link: LinkedObject | None = None
    permissions: dict[UID, set[str]] = {}
    storage_permissions: dict[UID, set[UID]] = {}
    ignored_batches: dict[UID, int] = {}
    object_sync_dates: dict[UID, DateTime] = {}
    errors: dict[UID, str] = {}

    # NOTE importing ServerDiff annotation with TYPE_CHECKING does not work here,
    # since typing.get_type_hints does not check for TYPE_CHECKING-imported types
    _previous_state_diff: Any = None

    __attr_searchable__ = ["created_at"]

    def _set_previous_state_diff(self) -> None:
        # relative
        from .diff_state import ServerDiff

        # Re-use ServerDiff to compare to previous state
        # Low = previous state, high = current state
        # NOTE No previous sync state means everything is new
        previous_state = self.previous_state or SyncState(
            server_uid=self.server_uid,
            server_name=self.server_name,
            server_side_type=self.server_side_type,
            syft_client_verify_key=self.syft_client_verify_key,
        )
        self._previous_state_diff = ServerDiff.from_sync_state(
            previous_state, self, _include_server_status=False, direction=None
        )

    def get_previous_state_diff(self) -> Any:
        if self._previous_state_diff is None:
            self._set_previous_state_diff()

        return self._previous_state_diff

    @property
    def previous_state(self) -> Optional["SyncState"]:
        if self.previous_state_link is not None:
            return self.previous_state_link.resolve
        return None

    @property
    def all_ids(self) -> set[UID]:
        return set(self.objects.keys())

    def get_status(self, uid: UID) -> str | None:
        previous_state_diff = self.get_previous_state_diff()
        if previous_state_diff is None:
            return None
        diff = previous_state_diff.obj_uid_to_diff.get(uid)

        if diff is None:
            return None
        return diff.status

    def add_objects(
        self, objects: list[SyncableSyftObject], context: AuthedServiceContext
    ) -> None:
        for obj in objects:
            if isinstance(obj.id, LineageID):
                self.objects[obj.id.id] = obj
            elif isinstance(obj.id, UID):
                self.objects[obj.id] = obj
            else:
                raise ValueError(f"Unsupported id type: {type(obj.id)}")

        # TODO might get slow with large states,
        # need to build dependencies every time to not have UIDs
        # in dependencies that are not in objects
        self._build_dependencies(context=context)

    def _build_dependencies(self, context: AuthedServiceContext) -> None:
        self.dependencies = {}

        all_ids = self.all_ids
        for obj in self.objects.values():
            if hasattr(obj, "get_sync_dependencies"):
                deps = obj.get_sync_dependencies(context=context)
                deps = [d.id for d in deps if d.id in all_ids]  # type: ignore
                # TODO: Why is this en check here? here?
                if len(deps):
                    self.dependencies[obj.id.id] = deps

    @property
    def rows(self) -> list[SyncStateRow]:
        result = []
        ids = set()

        previous_diff = self.get_previous_state_diff()
        if previous_diff is None:
            raise ValueError("No previous state to compare to")
        for batch in previous_diff.batches:
            # NOTE we re-use ServerDiff to compare to previous state,
            # low_obj is previous state, high_obj is current state
            diff = batch.root_diff

            # If there is no high object, it means it was deleted
            # and we don't need to show it in the table
            if diff.high_obj is None:
                continue
            if diff.object_id in ids:
                continue

            ids.add(diff.object_id)
            row = SyncStateRow(
                object=diff.high_obj,
                previous_object=diff.low_obj,
                current_state=diff.diff_side_str("high"),
                previous_state=diff.diff_side_str("low"),
                level=0,  # TODO add levels to table
                status=batch.status,
                last_sync_date=diff.last_sync_date,
            )
            result.append(row)
        return result

    def _repr_html_(self) -> str:
        prop_template = (
            "<p class='paragraph'><strong><span class='pr-8'>{}: </span></strong>{}</p>"
        )
        name_html = prop_template.format("name", self.server_name)
        if self.previous_state_link is not None:
            previous_state = self.previous_state_link.resolve
            delta = timedelta(
                seconds=self.created_at.utc_timestamp
                - previous_state.created_at.utc_timestamp
            )
            val = f"{td_format(delta)} ago"
            date_html = prop_template.format("last sync", val)
        else:
            date_html = prop_template.format("last sync", "not synced yet")

        repr = f"""
        <div class='syft-syncstate'>
            <p style="margin-bottom:16px;"></p>
            {name_html}
            {date_html}
        </div>
"""
        return repr + self.rows._repr_html_()
