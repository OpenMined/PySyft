# stdlib
import html
import textwrap
from typing import Any
from typing import ClassVar
from typing import Literal

# third party
from pydantic import model_validator
from rich import box
from rich.console import Console
from rich.console import Group
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from typing_extensions import Self

# relative
from ...client.client import SyftClient
from ...client.sync_decision import SyncDecision
from ...client.sync_decision import SyncDirection
from ...node.credentials import SyftVerifyKey
from ...types.datetime import DateTime
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SYFT_OBJECT_VERSION_2
from ...types.syft_object import SyftObject
from ...types.syft_object import short_uid
from ...types.syncable_object import SyncableSyftObject
from ...types.uid import LineageID
from ...types.uid import UID
from ...util import options
from ...util.colors import SURFACE
from ...util.fonts import FONT_CSS
from ...util.fonts import ITABLES_CSS
from ...util.notebook_ui.notebook_addons import ARROW_ICON
from ..action.action_object import ActionObject
from ..action.action_permissions import ActionObjectPermission
from ..action.action_permissions import ActionPermission
from ..action.action_permissions import StoragePermission
from ..code.user_code import UserCode
from ..code.user_code import UserCodeStatusCollection
from ..job.job_stash import Job
from ..log.log import SyftLog
from ..output.output_service import ExecutionOutput
from ..request.request import Request
from ..response import SyftError
from .sync_state import SyncState
from .sync_state import SyncView

sketchy_tab = "‎ " * 4


class AttrDiff(SyftObject):
    # version
    __canonical_name__ = "AttrDiff"
    __version__ = SYFT_OBJECT_VERSION_2
    attr_name: str
    low_attr: Any = None
    high_attr: Any = None

    def _repr_html_(self) -> str:
        return f"""{self.attr_name}:
    Low Side value: {self.low_attr}
    High Side value: {self.high_attr}
    """

    def __repr_side__(self, side: str) -> str:
        if side == "low":
            return recursive_attr_repr(self.low_attr)
        else:
            return recursive_attr_repr(self.high_attr)

    def _coll_repr_(self) -> dict[str, Any]:
        return {
            "attr name": self.attr_name,
            "low attr": html.escape(f"{self.low_attr}"),
            "high attr": html.escape(str(self.high_attr)),
        }


class ListDiff(AttrDiff):
    # version
    __canonical_name__ = "ListDiff"
    __version__ = SYFT_OBJECT_VERSION_2
    diff_ids: list[int] = []
    new_low_ids: list[int] = []
    new_high_ids: list[int] = []

    @property
    def is_empty(self) -> bool:
        return (
            len(self.diff_ids) == 0
            and len(self.new_low_ids) == 0
            and len(self.new_high_ids) == 0
        )

    @classmethod
    def from_lists(cls, attr_name: str, low_list: list, high_list: list) -> "ListDiff":
        diff_ids = []
        new_low_ids = []
        new_high_ids = []
        if len(low_list) != len(high_list):
            if len(low_list) > len(high_list):
                common_length = len(high_list)
                new_low_ids = list(range(common_length, len(low_list)))
            else:
                common_length = len(low_list)
                new_high_ids = list(range(common_length, len(high_list)))
        else:
            common_length = len(low_list)

        for i in range(common_length):
            if hasattr(low_list[i], "syft_eq"):
                if not low_list[i].syft_eq(high_list[i]):
                    diff_ids.append(i)
            elif low_list[i] != high_list[i]:
                diff_ids.append(i)

        change_diff = ListDiff(
            attr_name=attr_name,
            low_attr=low_list,
            high_attr=high_list,
            diff_ids=diff_ids,
            new_low_ids=new_low_ids,
            new_high_ids=new_high_ids,
        )
        return change_diff


def recursive_attr_repr(value_attr: list | dict | bytes, num_tabs: int = 0) -> str:
    new_num_tabs = num_tabs + 1

    if isinstance(value_attr, list):
        if len(value_attr) == 1:
            value_attr = value_attr[0]
        else:
            list_repr = "[\n"
            for elem in value_attr:
                list_repr += recursive_attr_repr(elem, num_tabs=num_tabs + 1) + "\n"
            list_repr += "]"
            return list_repr

    elif isinstance(value_attr, dict):
        dict_repr = "{\n"
        for key, elem in value_attr.items():
            dict_repr += f"{sketchy_tab * new_num_tabs}{key}: {str(elem)}\n"
        dict_repr += "}"
        return dict_repr

    elif isinstance(value_attr, bytes):
        value_attr = repr(value_attr)  # type: ignore
        if len(value_attr) > 50:
            value_attr = value_attr[:50] + "..."  # type: ignore

    if isinstance(value_attr, UID):
        value_attr = short_uid(value_attr)  # type: ignore

    return f"{sketchy_tab*num_tabs}{str(value_attr)}"


class ObjectDiff(SyftObject):  # StateTuple (compare 2 objects)
    # version
    __canonical_name__ = "ObjectDiff"
    __version__ = SYFT_OBJECT_VERSION_2
    low_obj: SyncableSyftObject | None = None
    high_obj: SyncableSyftObject | None = None
    low_node_uid: UID
    high_node_uid: UID
    low_permissions: list[str] = []
    high_permissions: list[str] = []
    low_storage_permissions: set[UID] = set()
    high_storage_permissions: set[UID] = set()
    low_status: str | None = None
    high_status: str | None = None
    last_sync_date_low: DateTime | None = None
    last_sync_dat_high: DateTime | None = None

    obj_type: type
    diff_list: list[AttrDiff] = []

    __repr_attrs__ = [
        "low_state",
        "high_state",
    ]

    def is_mock(self, side: str) -> bool:
        # An object is a mock object if it exists on both sides,
        # and has no storage permissions on `side`
        # NOTE both sides must have the objects, else it is a new object.
        # New+mock objects do not appear naturally, but if they do we
        # want them to show up.
        if side == "low":
            obj = self.low_obj
            other_obj = self.high_obj
            permissions = self.low_storage_permissions
            node_uid = self.low_node_uid
        elif side == "high":
            obj = self.high_obj
            other_obj = self.low_obj
            permissions = self.high_storage_permissions
            node_uid = self.high_node_uid
        else:
            raise ValueError("Invalid side")

        if obj is None or other_obj is None:
            return False

        return node_uid not in permissions

    @classmethod
    def from_objects(
        cls,
        low_obj: SyncableSyftObject | None,
        high_obj: SyncableSyftObject | None,
        low_status: str | None,
        high_status: str | None,
        low_permissions: set[str],
        high_permissions: set[str],
        low_storage_permissions: set[UID],
        high_storage_permissions: set[UID],
        low_node_uid: UID,
        high_node_uid: UID,
        last_sync_date_low: DateTime | None = None,
        last_sync_date_high: DateTime | None = None,
    ) -> "ObjectDiff":
        if low_obj is None and high_obj is None:
            raise ValueError("Both low and high objects are None")
        obj_type = type(low_obj if low_obj is not None else high_obj)

        res = cls(
            low_obj=low_obj,
            high_obj=high_obj,
            low_status=low_status,
            high_status=high_status,
            obj_type=obj_type,
            low_node_uid=low_node_uid,
            high_node_uid=high_node_uid,
            low_permissions=low_permissions,
            high_permissions=high_permissions,
            low_storage_permissions=low_storage_permissions,
            high_storage_permissions=high_storage_permissions,
            last_sync_date_low=last_sync_date_low,
            last_sync_date_high=last_sync_date_high,
        )

        if (
            low_obj is None
            or high_obj is None
            or (res.is_mock("low") and high_status == "SAME")
            or (res.is_mock("high") and low_status == "SAME")
        ):
            diff_list = []
        else:
            diff_list = low_obj.syft_get_diffs(high_obj)

        res.diff_list = diff_list
        return res

    def __hash__(self) -> int:
        return hash(self.object_id) + hash(self.low_obj) + hash(self.high_obj)

    @property
    def last_sync_date(self) -> DateTime | None:
        last_sync_low = self.last_sync_date_low if self.low_obj is not None else None
        last_sync_high = self.last_sync_dat_high if self.high_obj is not None else None

        if last_sync_low is None:
            return last_sync_high
        elif last_sync_high is None:
            return last_sync_low
        else:
            return max(last_sync_low, last_sync_high)

    @property
    def status(self) -> Literal["NEW", "SAME", "MODIFIED"]:
        if self.low_obj is None or self.high_obj is None:
            return "NEW"
        if len(self.diff_list) == 0:
            return "SAME"
        return "MODIFIED"

    @property
    def object_id(self) -> UID:
        uid: UID | LineageID = (
            self.low_obj.id if self.low_obj is not None else self.high_obj.id  # type: ignore
        )
        if isinstance(uid, LineageID):
            return uid.id
        return uid

    @property
    def non_empty_object(self) -> SyftObject | None:
        return self.low_obj or self.high_obj

    @property
    def object_type(self) -> str:
        return self.obj_type.__name__

    @property
    def high_state(self) -> str:
        return self.state_str("high")

    @property
    def low_state(self) -> str:
        return self.state_str("low")

    @property
    def object_uid(self) -> UID:
        return self.low_obj.id if self.low_obj is not None else self.high_obj.id  # type: ignore

    def repr_attr_diffstatus_dict(self) -> dict:
        # relative
        from .resolve_widget import DiffStatus

        low_attrs = self.repr_attr_dict("low")
        high_attrs = self.repr_attr_dict("high")
        all_attrs = set(low_attrs.keys()) | set(high_attrs.keys())

        res = {}
        for attr in all_attrs:
            value_low = low_attrs.get(attr, None)
            value_high = high_attrs.get(attr, None)

            if value_low is None or value_high is None:
                res[attr] = DiffStatus.NEW
            elif value_low != value_high:
                res[attr] = DiffStatus.MODIFIED
            else:
                res[attr] = DiffStatus.SAME
        return res

    def repr_attr_dict(self, side: str) -> dict[str, Any]:
        obj = self.low_obj if side == "low" else self.high_obj
        if isinstance(obj, ActionObject):
            return {"value": obj.syft_action_data_cache}
        repr_attrs = getattr(obj, "__repr_attrs__", [])
        res = {}
        for attr in repr_attrs:
            value = getattr(obj, attr)
            res[attr] = value
        return res

    def diff_attributes_str(self, side: str) -> str:
        obj = self.low_obj if side == "low" else self.high_obj

        if obj is None:
            return ""

        repr_attrs = getattr(obj, "__repr_attrs__", [])
        if self.status == "SAME":
            repr_attrs = repr_attrs[:3]

        if self.status in {"SAME", "NEW"}:
            attrs_str = ""
            for attr in repr_attrs:
                value = getattr(obj, attr)
                attrs_str += f"{attr}: {recursive_attr_repr(value)}\n"
            return attrs_str

        elif self.status == "MODIFIED":
            attrs_str = ""
            for diff in self.diff_list:
                attrs_str += f"{diff.attr_name}: {diff.__repr_side__(side)}\n"
            return attrs_str
        else:
            raise ValueError("")

    def diff_side_str(self, side: str) -> str:
        obj = self.low_obj if side == "low" else self.high_obj
        if obj is None:
            return ""
        res = f"{self.obj_type.__name__.upper()} #{short_uid(obj.id)}\n"
        res += self.diff_attributes_str(side)
        return res

    def state_str(self, side: str) -> str:
        other_obj: SyftObject | None = None
        if side == "high":
            obj = self.high_obj
            other_obj = self.low_obj
        else:
            obj = self.low_obj
            other_obj = self.high_obj

        if obj is None:
            return "-"
        if self.status == "SAME":
            return f"SAME\n{self.obj_type.__name__}"

        if isinstance(obj, ActionObject):
            return obj.__repr__()

        if other_obj is None:  # type: ignore[unreachable]
            attrs_str = ""
            attrs = getattr(obj, "__repr_attrs__", [])
            for attr in attrs:
                value = getattr(obj, attr)
                attrs_str += f"{sketchy_tab}{attr} = {recursive_attr_repr(value)}\n"
            attrs_str = attrs_str[:-1]
            return f"NEW\n\nclass {self.object_type}:\n{attrs_str}"

        attr_text = f"DIFF\nclass {self.object_type}:\n"
        for diff in self.diff_list:
            # TODO
            attr_text += (
                f"{sketchy_tab}{diff.attr_name}={diff.__repr_side__(side)}," + "\n"
            )
        if len(self.diff_list) > 0:
            attr_text = attr_text[:-2]

        return attr_text

    def get_obj(self) -> SyftObject | None:
        if self.status == "NEW":
            return self.low_obj if self.low_obj is not None else self.high_obj
        else:
            raise ValueError("Cannot get object from a diff that is not new")

    def _coll_repr_(self) -> dict[str, Any]:
        low_state = f"{self.status}\n{self.diff_side_str('low')}"
        high_state = f"{self.status}\n{self.diff_side_str('high')}"
        return {
            "low_state": html.escape(low_state),
            "high_state": html.escape(high_state),
        }

    def _repr_html_(self) -> str:
        if self.low_obj is None and self.high_obj is None:
            return SyftError(message="Something broke")

        base_str = f"""
        <style>
        {FONT_CSS}
        .syft-dataset {{color: {SURFACE[options.color_theme]};}}
        .syft-dataset h3,
        .syft-dataset p
            {{font-family: 'Open Sans';}}
            {ITABLES_CSS}
        </style>
        <div class='syft-diff'>
        """

        obj_repr: str
        attr_text: str
        if self.low_obj is None:
            if hasattr(self.high_obj, "_repr_html_"):
                obj_repr = self.high_obj._repr_html_()  # type: ignore
            elif hasattr(self.high_obj, "_inner_repr"):
                obj_repr = self.high_obj._inner_repr()  # type: ignore
            else:
                obj_repr = self.__repr__()
            attr_text = (
                f"""
    <h3>{self.object_type} ObjectDiff (New {self.object_type}  on the High Side):</h3>
    """
                + obj_repr
            )

        elif self.high_obj is None:
            if hasattr(self.low_obj, "_repr_html_"):
                obj_repr = self.low_obj._repr_html_()  # type: ignore
            elif hasattr(self.low_obj, "_inner_repr"):
                obj_repr = self.low_obj._inner_repr()  # type: ignore
            else:
                obj_repr = self.__repr__()
            attr_text = (
                f"""
    <h3>{self.object_type} ObjectDiff (New {self.object_type}  on the High Side):</h3>
    """
                + obj_repr
            )

        elif self.status == "SAME":
            obj_repr = "No changes between low side and high side"
        else:
            obj_repr = ""
            for diff in self.diff_list:
                obj_repr += diff.__repr__() + "<br>"

            obj_repr = obj_repr.replace("\n", "<br>")
            # print("New lines", res)

        attr_text = f"<h3>{self.object_type} ObjectDiff:</h3>\n{obj_repr}"
        return base_str + attr_text

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[{self.obj_type.__name__}](#{str(self.object_id)})"


def _wrap_text(text: str, width: int, indent: int = 4) -> str:
    """Wrap text, preserving existing line breaks"""
    return "\n".join(
        [
            "\n".join(
                textwrap.wrap(
                    line,
                    width,
                    break_long_words=False,
                    replace_whitespace=False,
                    subsequent_indent=" " * indent,
                )
            )
            for line in text.splitlines()
            if line.strip() != ""
        ]
    )


class ObjectDiffBatch(SyftObject):
    __canonical_name__ = "DiffHierarchy"
    __version__ = SYFT_OBJECT_VERSION_2
    LINE_LENGTH: ClassVar[int] = 100
    INDENT: ClassVar[int] = 4
    ORDER: ClassVar[dict] = {"low": 0, "high": 1}

    __syft_include_id_coll_repr__ = False

    low_node_uid: UID
    high_node_uid: UID
    user_verify_key_low: SyftVerifyKey
    user_verify_key_high: SyftVerifyKey

    # Diffs are ordered in depth-first order,
    # the first diff is the root of the hierarchy
    global_diffs: dict[UID, ObjectDiff]
    global_roots: list[UID]

    hierarchy_levels: list[int]
    dependencies: dict[UID, list[UID]] = {}
    dependents: dict[UID, list[UID]] = {}
    decision: SyncDecision | None = None
    root_diff: ObjectDiff
    sync_direction: SyncDirection | None

    def walk_graph(
        self,
        deps: dict[UID, list[UID]],
        include_roots: bool = False,
        include_batch_root: bool = True,
    ) -> list[ObjectDiff]:
        root_id = self.root_diff.object_id
        result = [root_id]
        unvisited = [root_id]
        global_roots = [x for x in self.global_roots if x is not root_id]
        roots = []

        while len(unvisited):
            # Do we update this in the terminal case
            new_nodes = []
            for node in unvisited:
                if node in global_roots:
                    roots.append(node)
                else:
                    new_nodes += deps.get(node, [])

            new_nodes = [n for n in new_nodes if n not in result]
            unvisited = new_nodes
            result += unvisited

        if include_roots:
            result += roots

        if not include_batch_root:
            result.remove(root_id)

        return [self.global_diffs[r] for r in set(result)]

    @property
    def target_node_uid(self) -> UID:
        if self.sync_direction is None:
            raise ValueError("no direction specified")
        if self.sync_direction == SyncDirection.LOW_TO_HIGH:
            return self.high_node_uid
        else:
            return self.low_node_uid

    @property
    def target_verify_key(self) -> SyftVerifyKey:
        if self.sync_direction is None:
            raise ValueError("no direction specified")
        if self.sync_direction == SyncDirection.LOW_TO_HIGH:
            return self.user_verify_key_high
        else:
            return self.user_verify_key_low

    def get_dependencies(
        self,
        include_roots: bool = False,
        include_batch_root: bool = True,
    ) -> list[ObjectDiff]:
        return self.walk_graph(
            deps=self.dependencies,
            include_roots=include_roots,
            include_batch_root=include_batch_root,
        )

    @property
    def status(self) -> str:
        if self.root_diff.status == "NEW":
            return "NEW"

        batch_statuses = [
            diff.status for diff in self.get_dependents(include_roots=False)
        ]
        if all(status == "SAME" for status in batch_statuses):
            return "SAME"

        return "MODIFIED"

    @property
    def is_unchanged(self) -> bool:
        return self.status == "SAME"

    def get_dependents(
        self, include_roots: bool = False, include_batch_root: bool = True
    ) -> list[ObjectDiff]:
        return self.walk_graph(
            deps=self.dependents,
            include_roots=include_roots,
            include_batch_root=include_batch_root,
        )

    def __hash__(self) -> int:
        diffs = self.get_dependents(include_roots=False)
        return sum(hash(x) for x in diffs)

    @property
    def root_id(self) -> UID:
        return self.root_diff.object_id

    @property
    def root_type(self) -> type:
        return self.root_diff.obj_type

    @property
    def is_ignored(self) -> bool:
        return self.decision == SyncDecision.ignore

    @property
    def is_skipped(self) -> bool:
        return self.decision == SyncDecision.skip

    @classmethod
    def from_dependencies(
        cls,
        root_uid: UID,
        obj_dependencies: dict[UID, list[UID]],
        obj_uid_to_diff: dict[UID, ObjectDiff],
        root_ids: list[UID],
        low_node_uid: UID,
        high_node_uid: UID,
        user_verify_key_low: SyftVerifyKey,
        user_verify_key_high: SyftVerifyKey,
        sync_direction: SyncDirection,
    ) -> "ObjectDiffBatch":
        def _build_hierarchy_helper(
            uid: UID, level: int = 0, visited: set | None = None
        ) -> list:
            visited = visited if visited is not None else set()

            if uid in visited:
                return []

            result = [(uid, level)]
            visited.add(uid)
            if uid in obj_dependencies:
                deps = obj_dependencies[uid]
                for dep_uid in obj_dependencies[uid]:
                    if dep_uid not in visited:
                        # NOTE we pass visited + deps to recursive calls, to have
                        # all objects at the highest level in the hierarchy
                        # Example:
                        # ExecutionOutput
                        # -- Job
                        # ---- Result
                        # -- Result
                        # We want to omit Job.Result, because it's already in ExecutionOutput.Result
                        result.extend(
                            _build_hierarchy_helper(
                                uid=dep_uid,
                                level=level + 1,
                                visited=visited | set(deps) - {dep_uid},
                            )
                        )
            return result

        batch_uids = _build_hierarchy_helper(root_uid)
        # levels in the tree that we create
        levels = [level for _, level in batch_uids]

        batch_uids = {uid for uid, _ in batch_uids}  # type: ignore
        batch_dependencies = {
            uid: [d for d in obj_dependencies.get(uid, []) if d in batch_uids]
            for uid in batch_uids
        }
        return cls(
            global_diffs=obj_uid_to_diff,
            global_roots=root_ids,
            hierarchy_levels=levels,
            dependencies=batch_dependencies,
            root_diff=obj_uid_to_diff[root_uid],
            low_node_uid=low_node_uid,
            high_node_uid=high_node_uid,
            user_verify_key_low=user_verify_key_low,
            user_verify_key_high=user_verify_key_high,
            sync_direction=sync_direction,
        )

    def flatten_visual_hierarchy(self) -> list[ObjectDiff]:
        def flatten_dict(d: dict) -> list:
            if len(d) == 0:
                return []
            else:
                result = []
                for diff, child in d.items():
                    result.append(diff)
                    result += flatten_dict(child)
                return result

        return flatten_dict(self.get_visual_hierarchy())

    def _repr_html_(self) -> str:
        diffs = self.flatten_visual_hierarchy()

        return f"""
<h2> ObjectBatchDiff </h2>
{diffs._repr_html_()}
"""

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
        no_obj_html = "<p class='diff-state-no-obj'>No object detected</p>"
        if self.root_diff.low_obj is None:
            low_html = no_obj_html
        else:
            low_html = SyncView(object=self.root_diff.low_obj).summary_html()

        if self.root_diff.high_obj is None:
            high_html = no_obj_html
        else:
            high_html = SyncView(object=self.root_diff.high_obj).summary_html()

        return {
            "Merge status": self.status_badge(),
            "Public Sync State": low_html,
            "Private sync state": high_html,
        }

    @property
    def visual_hierarchy(self) -> tuple[type, dict]:
        # Returns
        root_obj = (
            self.root.low_obj if self.root.low_obj is not None else self.root.high_obj
        )
        if isinstance(root_obj, Request):
            return Request, {
                Request: [UserCode],
            }
        elif isinstance(root_obj, UserCode):
            return UserCode, {  # type: ignore
                UserCode: [UserCodeStatusCollection, UserCode],
            }
        elif isinstance(root_obj, Job):
            return UserCode, {  # type: ignore
                UserCode: [ExecutionOutput, UserCode],
                ExecutionOutput: [Job],
                Job: [ActionObject, SyftLog, Job],
            }
        else:
            raise ValueError(f"Unknown root type: {self.root.obj_type}")

    @model_validator(mode="after")
    def make_dependents(self) -> Self:
        dependents: dict = {}
        for parent, children in self.dependencies.items():
            for child in children:
                dependents[child] = dependents.get(child, []) + [parent]
        self.dependents = dependents
        return self

    @property
    def root(self) -> ObjectDiff:
        return self.root_diff

    def __repr__(self) -> str:
        return f"""{self.hierarchy_str('low')}

{self.hierarchy_str('high')}
"""

    def _repr_markdown_(self, wrap_as_python: bool = True, indent: int = 0) -> str:
        return ""  # Turns off the _repr_markdown_ of SyftObject

    def _get_visual_hierarchy(
        self, node: ObjectDiff, visited: set[UID] | None = None
    ) -> dict[ObjectDiff, dict]:
        visited = visited if visited is not None else set()
        visited.add(node.object_id)

        _, child_types_map = self.visual_hierarchy
        child_types = child_types_map.get(node.obj_type, [])
        dep_ids = self.dependencies.get(node.object_id, []) + self.dependents.get(
            node.object_id, []
        )

        result = {}
        for child_type in child_types:
            children = [
                n
                for n in self.global_diffs.values()
                if n.object_id in dep_ids
                and isinstance(n.low_obj or n.high_obj, child_type)
            ]
            for child in children:
                if child.object_id not in visited:
                    result[child] = self._get_visual_hierarchy(child, visited=visited)

        return result

    @property
    def visual_root(self) -> ObjectDiff:
        dependecies: list[ObjectDiff] = self.get_dependencies(include_roots=True)
        visual_root_type = self.visual_hierarchy[0]

        visual_roots = [
            diff
            for diff in dependecies
            if isinstance(diff.low_obj or diff.high_obj, visual_root_type)
        ]
        if not len(visual_roots):
            raise ValueError("No visual root found")

        return visual_roots[0]

    @property
    def user_code_high(self) -> UserCode | None:
        """return the user code of the high side of this batch, if it exists"""
        user_codes_high: list[UserCode] = [
            diff.high_obj
            for diff in self.get_dependencies(include_roots=True)
            if isinstance(diff.high_obj, UserCode)
        ]

        if len(user_codes_high) == 0:
            user_code_high = None
        else:
            # NOTE we can always assume the first usercode is
            # not a nested code, because diffs are sorted in depth-first order
            user_code_high = user_codes_high[0]
        return user_code_high

    def get_visual_hierarchy(self) -> dict[ObjectDiff, dict]:
        visual_root = self.visual_root
        return {visual_root: self._get_visual_hierarchy(self.visual_root)}  # type: ignore

    def _get_obj_str(self, diff_obj: ObjectDiff, level: int, side: str) -> str:
        obj = diff_obj.low_obj if side == "low" else diff_obj.high_obj
        if obj is None:
            return ""
        indent = " " * level * self.INDENT
        obj_str = diff_obj.diff_side_str(side)
        obj_str = _wrap_text(obj_str, width=self.LINE_LENGTH - len(indent))

        line_prefix = indent + f"―――― {diff_obj.status} "
        line = "―" * (self.LINE_LENGTH - len(line_prefix))
        return f"""{line_prefix}{line}

{textwrap.indent(obj_str, indent)}

"""

    def hierarchy_str(self, side: str) -> str:
        def _hierarchy_str_recursive(tree: dict, level: int) -> str:
            result = ""
            for node, children in tree.items():
                result += self._get_obj_str(node, level, side)
                result += _hierarchy_str_recursive(children, level + 1)
            return result

        visual_hierarchy = self.get_visual_hierarchy()
        res = _hierarchy_str_recursive(visual_hierarchy, 0)
        if res == "":
            res = f"No {side} side changes."
        return f"""{side.upper()} SIDE STATE:

{res}"""


class IgnoredBatchView(SyftObject):
    __canonical_name__ = "IgnoredBatchView"
    __version__ = SYFT_OBJECT_VERSION_1
    batch: ObjectDiffBatch
    other_batches: list[ObjectDiffBatch]

    def _coll_repr_(self) -> dict[str, Any]:
        return self.batch._coll_repr_()

    def _repr_html_(self) -> str:
        return self.batch._repr_html_()

    def stage_change(self) -> None:
        self.batch.decision = None
        required_dependencies = {
            d.object_id for d in self.batch.get_dependencies(include_roots=True)
        }

        for other_batch in self.other_batches:
            if (
                other_batch.decision == SyncDecision.ignore
                and other_batch.root_id in required_dependencies
            ):
                print(f"ignoring other batch ({other_batch.root_type.__name__})")
                other_batch.decision = None


class NodeDiff(SyftObject):
    __canonical_name__ = "NodeDiff"
    __version__ = SYFT_OBJECT_VERSION_2

    low_node_uid: UID
    high_node_uid: UID
    user_verify_key_low: SyftVerifyKey
    user_verify_key_high: SyftVerifyKey
    obj_uid_to_diff: dict[UID, ObjectDiff] = {}
    obj_dependencies: dict[UID, list[UID]] = {}
    batches: list[ObjectDiffBatch] = []
    low_state: SyncState
    high_state: SyncState
    direction: SyncDirection | None

    def __getitem__(self, idx: Any) -> ObjectDiffBatch:
        return self.batches[idx]

    @property
    def ignored_changes(self) -> list[IgnoredBatchView]:
        ignored_batches = [b for b in self.batches if b.decision == SyncDecision.ignore]
        result = []
        for ignored_batch in ignored_batches:
            other_batches = [b for b in self.batches if b is not ignored_batch]
            result.append(
                IgnoredBatchView(batch=ignored_batch, other_batches=other_batches)
            )
        return result

    @classmethod
    def from_sync_state(
        cls: type["NodeDiff"],
        low_state: SyncState,
        high_state: SyncState,
        direction: SyncDirection,
        _include_node_status: bool = False,
    ) -> "NodeDiff":
        obj_uid_to_diff = {}
        for obj_id in set(low_state.objects.keys()) | set(high_state.objects.keys()):
            low_obj = low_state.objects.get(obj_id, None)
            high_obj = high_state.objects.get(obj_id, None)

            low_permissions = low_state.permissions.get(obj_id, set())
            high_permissions = high_state.permissions.get(obj_id, set())

            low_storage_permissions = low_state.storage_permissions.get(obj_id, set())
            high_storage_permissions = high_state.storage_permissions.get(obj_id, set())

            last_sync_date_low = low_state.object_sync_dates.get(obj_id, None)
            last_sync_date_high = high_state.object_sync_dates.get(obj_id, None)

            if _include_node_status:
                low_status = low_state.get_status(obj_id)
                high_status = high_state.get_status(obj_id)
            else:
                low_status = "NEW"
                high_status = "NEW"

            diff = ObjectDiff.from_objects(
                low_obj=low_obj,
                high_obj=high_obj,
                low_status=low_status,
                high_status=high_status,
                low_permissions=low_permissions,
                high_permissions=high_permissions,
                low_storage_permissions=low_storage_permissions,
                high_storage_permissions=high_storage_permissions,
                low_node_uid=low_state.node_uid,
                high_node_uid=high_state.node_uid,
                last_sync_date_low=last_sync_date_low,
                last_sync_date_high=last_sync_date_high,
            )
            obj_uid_to_diff[diff.object_id] = diff

        obj_dependencies = NodeDiff.dependencies_from_states(low_state, high_state)
        batches = NodeDiff.hierarchies(
            low_state,
            high_state,
            obj_dependencies,
            obj_uid_to_diff,
            direction=direction,
        )

        # TODO: Check if high and low ignored batches are the same else error
        previously_ignored_batches = low_state.ignored_batches
        NodeDiff.apply_previous_ignore_state(batches, previously_ignored_batches)

        return cls(
            low_node_uid=low_state.node_uid,
            high_node_uid=high_state.node_uid,
            user_verify_key_low=low_state.syft_client_verify_key,
            user_verify_key_high=high_state.syft_client_verify_key,
            obj_uid_to_diff=obj_uid_to_diff,
            obj_dependencies=obj_dependencies,
            batches=batches,
            low_state=low_state,
            high_state=high_state,
            direction=direction,
        )

    @staticmethod
    def apply_previous_ignore_state(
        batches: list[ObjectDiffBatch], previously_ignored_batches: dict[UID, int]
    ) -> None:
        """Loop through all ignored batches in syncstate. If batch did not change, set to ignored
        If another batch needs to exist in order to accept that changed batch: also unignore
        e.g. if a job changed, also unignore the usercode"""

        for root_id, batch_hash in previously_ignored_batches.items():
            for batch in batches:
                if batch.root_id == root_id:
                    if hash(batch) == batch_hash:
                        batch.decision = SyncDecision.ignore
                    else:
                        print(
                            f"""A batch with type {batch.root_type.__name__} was previously ignored but has changed
It will be available for review again."""
                        )
                        # batch has changed, so unignore
                        batch.decision = None
                        # then we also set the dependent batches to unignore
                        # currently we dont do this recusively
                        required_dependencies = {
                            d.object_id
                            for d in batch.get_dependencies(include_roots=True)
                        }

                        for other_batch in batches:
                            if other_batch is not batch:
                                other_batch_root_id = {other_batch.root_id}
                                # if there is overlap
                                if len(required_dependencies & other_batch_root_id):
                                    other_batch.decision = None

    @staticmethod
    def dependencies_from_states(
        low_state: SyncState, high_state: SyncState
    ) -> dict[UID, list[UID]]:
        dependencies = {}
        all_parents = set(low_state.dependencies.keys()) | set(
            high_state.dependencies.keys()
        )
        for parent in all_parents:
            low_deps = low_state.dependencies.get(parent, [])
            high_deps = high_state.dependencies.get(parent, [])
            dependencies[parent] = list(set(low_deps) | set(high_deps))
        return dependencies

    @property
    def diffs(self) -> list[ObjectDiff]:
        diffs_depthfirst = [
            diff
            for hierarchy in self.batches
            for diff in hierarchy.get_dependents(include_roots=False)
        ]
        # deduplicate
        diffs = []
        ids = set()
        for diff in diffs_depthfirst:
            if diff.object_id not in ids:
                diffs.append(diff)
                ids.add(diff.object_id)
        return diffs

    def _repr_markdown_(self) -> None:  # type: ignore
        return None

    def _repr_html_(self) -> Any:
        n = len(self.batches)
        if self.direction == SyncDirection.LOW_TO_HIGH:
            name1 = "Public Node"
            name2 = "Private Node"
        else:
            name1 = "Private Node"
            name2 = "Public Node"
        repr_html = f"""
        <p style="margin-bottom:16px;"></p>
        <div class="diff-state-intro">Comparing sync states</div>
        <p style="margin-bottom:16px;"></p>
        <div class="diff-state-header"><span>{name1}</span> {ARROW_ICON} <span>{name2}</span></div>
        <p style="margin-bottom:16px;"></p>
        <div class="diff-state-sub-header"> This would sync <span class="diff-state-orange-text">{n} batches</span> from <i>{name1}</i> to <i>{name2}</i></div>
        """  # noqa: E501
        repr_html = repr_html.replace("\n", "")

        return repr_html + self.batches._repr_html_()

    @staticmethod
    def _sort_batches(hierarchies: list[ObjectDiffBatch]) -> list[ObjectDiffBatch]:
        without_usercode = []
        grouped_by_usercode: dict[UID, list[ObjectDiffBatch]] = {}
        for hierarchy in hierarchies:
            has_usercode = False
            for diff in hierarchy.get_dependencies(include_roots=True):
                obj = diff.low_obj if diff.low_obj is not None else diff.high_obj
                if isinstance(obj, UserCode):
                    usercode_id = obj.id
                    if usercode_id not in grouped_by_usercode:
                        grouped_by_usercode[usercode_id] = []
                    grouped_by_usercode[usercode_id].append(hierarchy)
                    has_usercode = True
                    break
            if not has_usercode:
                without_usercode.append(hierarchy)

        # Order of hierarchies, by root object type
        hierarchy_order = [UserCode, Request, Job]
        # Sort group by hierarchy_order, then by root object id
        for hierarchy_group in grouped_by_usercode.values():
            hierarchy_group.sort(
                key=lambda x: (
                    hierarchy_order.index(x.root.obj_type),
                    x.root.object_id,
                )
            )

        # sorted = sorted groups + without_usercode
        sorted_hierarchies = []
        for grp in grouped_by_usercode.values():
            sorted_hierarchies.extend(grp)
        sorted_hierarchies.extend(without_usercode)
        return sorted_hierarchies

    @staticmethod
    def hierarchies(
        low_sync_state: SyncState,
        high_sync_state: SyncState,
        obj_dependencies: dict[UID, list[UID]],
        obj_uid_to_diff: dict[UID, ObjectDiff],
        direction: SyncDirection,
    ) -> list[ObjectDiffBatch]:
        batches = []
        root_ids = []

        for diff in obj_uid_to_diff.values():
            diff_obj = diff.low_obj if diff.low_obj is not None else diff.high_obj
            if isinstance(diff_obj, Request):
                root_ids.append(diff.object_id)
            elif isinstance(diff_obj, Job) and diff_obj.parent_job_id is None:  # type: ignore
                root_ids.append(diff.object_id)  # type: ignore
            elif isinstance(diff_obj, UserCode):
                # TODO: Figure out nested user codes, do we even need that?
                root_ids.append(diff.object_id)  # type: ignore

        for root_uid in root_ids:
            batch = ObjectDiffBatch.from_dependencies(
                root_uid,
                obj_dependencies,
                obj_uid_to_diff,
                root_ids,
                low_sync_state.node_uid,
                high_sync_state.node_uid,
                low_sync_state.syft_client_verify_key,
                high_sync_state.syft_client_verify_key,
                sync_direction=direction,
            )
            batches.append(batch)

        hierarchies_sorted = NodeDiff._sort_batches(batches)
        return hierarchies_sorted

    @property
    def is_same(self) -> bool:
        return all(object_diff.status == "SAME" for object_diff in self.diffs)


class SyncInstruction(SyftObject):
    __canonical_name__ = "SyncDecision"
    __version__ = SYFT_OBJECT_VERSION_2

    diff: ObjectDiff
    decision: SyncDecision | None
    new_permissions_lowside: list[ActionObjectPermission]
    new_storage_permissions_lowside: list[StoragePermission]
    new_storage_permissions_highside: list[StoragePermission]
    unignore: bool = False
    mockify: bool

    @classmethod
    def from_widget_state(
        cls,
        sync_direction: SyncDirection,
        widget: Any,
        decision: SyncDecision,
        share_to_user: SyftVerifyKey | None,
    ) -> Self:
        # read widget state
        diff = widget.diff
        new_permissions_low_side = []

        # read permissions
        if sync_direction == SyncDirection.HIGH_TO_LOW:
            if widget.share_private_data or diff.object_type == "Job":
                if share_to_user is None:
                    raise ValueError("empty to user to share with")
                new_permissions_low_side = [
                    ActionObjectPermission(
                        uid=widget.diff.object_id,
                        permission=ActionPermission.READ,
                        credentials=share_to_user,  # type: ignore
                    )
                ]

        # mockify
        mockify = widget.mockify

        # storage permissions
        new_storage_permissions = []

        if sync_direction == SyncDirection.HIGH_TO_LOW:
            # TODO: apply storage permissions on both ends
            if not mockify:
                new_storage_permissions.append(
                    StoragePermission(uid=diff.object_id, node_uid=diff.low_node_uid)
                )
        elif sync_direction == SyncDirection.LOW_TO_HIGH:
            new_storage_permissions.append(
                StoragePermission(uid=diff.object_id, node_uid=diff.high_node_uid)
            )

        return cls(
            diff=diff,
            decision=decision,
            new_permissions_lowside=new_permissions_low_side,
            new_storage_permissions_lowside=new_storage_permissions,
            new_storage_permissions_highside=new_storage_permissions,
            mockify=mockify,
        )


class ResolvedSyncState(SyftObject):
    __canonical_name__ = "SyncUpdate"
    __version__ = SYFT_OBJECT_VERSION_2

    node_uid: UID
    create_objs: list[SyncableSyftObject] = []
    update_objs: list[SyncableSyftObject] = []
    delete_objs: list[SyftObject] = []
    new_permissions: list[ActionObjectPermission] = []
    new_storage_permissions: list[StoragePermission] = []
    ignored_batches: dict[UID, int] = {}  # batch root uid -> hash of the batch
    unignored_batches: set[UID] = (
        set()
    )  # NOTE: using '{}' as default value does not work here
    alias: str

    @classmethod
    def from_client(cls, client: SyftClient) -> "ResolvedSyncState":
        alias: str = client.metadata.node_side_type  # type: ignore
        if alias not in ["low", "high"]:
            raise ValueError(
                "can only create resolved sync state for high, low side deployments"
            )
        return cls(node_uid=client.id, alias=alias)

    def add_ignored(self, batch: ObjectDiffBatch) -> None:
        self.ignored_batches[batch.root_id] = hash(batch)

    def add_unignored(self, root_id: UID) -> None:
        self.unignored_batches.add(root_id)

    def add_sync_instruction(self, sync_instruction: SyncInstruction) -> None:
        if (
            sync_instruction.decision == SyncDecision.ignore
            or sync_instruction.decision == SyncDecision.skip
        ):
            return
        diff = sync_instruction.diff

        if sync_instruction.unignore:
            self.unignored_batches.add(sync_instruction.batch_diff.root_id)

        if diff.status == "SAME":
            return

        my_obj = diff.low_obj if self.alias == "low" else diff.high_obj
        other_obj = diff.low_obj if self.alias == "high" else diff.high_obj

        if other_obj is not None and sync_instruction.mockify:
            other_obj = other_obj.create_shareable_sync_copy(mock=True)

        if (
            sync_instruction.decision and sync_instruction.decision.value != self.alias
        ):  # chose for the other
            if diff.status == "MODIFIED":
                # keep IDs comparison here, otherwise it will break with actionobjects
                if other_obj.id not in [x.id for x in self.update_objs]:  # type: ignore
                    self.update_objs.append(other_obj)

            elif diff.status == "NEW":
                if my_obj is None:
                    # keep IDs comparison here, otherwise it will break with actionobjects
                    if other_obj.id not in [x.id for x in self.create_objs]:  # type: ignore
                        self.create_objs.append(other_obj)

                elif other_obj is None:
                    # keep IDs comparison here, otherwise it will break with actionobjects
                    if my_obj.id not in [x.id for x in self.delete_objs]:
                        self.delete_objs.append(my_obj)

        if self.alias == "low":
            self.new_permissions.extend(sync_instruction.new_permissions_lowside)
            self.new_storage_permissions.extend(
                sync_instruction.new_storage_permissions_lowside
            )
        elif self.alias == "high":
            self.new_storage_permissions.extend(
                sync_instruction.new_storage_permissions_highside
            )
        else:
            raise ValueError("Invalid alias")

    @property
    def is_empty(self) -> bool:
        return (
            len(self.create_objs) == 0
            and len(self.update_objs) == 0
            and len(self.delete_objs) == 0
            and len(self.new_permissions) == 0
            and len(self.new_storage_permissions) == 0
        )

    def __repr__(self) -> str:
        return (
            f"ResolvedSyncState(\n"
            f"  create_objs={self.create_objs},\n"
            f"  update_objs={self.update_objs},\n"
            f"  delete_objs={self.delete_objs}\n"
            f"  new_permissions={self.new_permissions}\n"
            f"  new_storage_permissions={self.new_storage_permissions}\n"
            f"  ignored_batches={list(self.ignored_batches.keys())}\n"
            f")"
        )


def display_diff_object(obj_state: str | None) -> Panel:
    if obj_state is None:
        return Panel(Markdown("None"), box=box.ROUNDED, expand=False)
    return Panel(
        Markdown(f"```python\n{obj_state}\n```", code_theme="default"),
        box=box.ROUNDED,
        expand=False,
    )


def display_diff_hierarchy(diff_hierarchy: list[tuple[ObjectDiff, int]]) -> None:
    console = Console()

    for diff, level in diff_hierarchy:
        title = f"{diff.obj_type.__name__}({diff.object_id}) - State: {diff.status}"

        low_side_panel = display_diff_object(diff.low_state if diff.low_obj else None)
        low_side_panel.title = "Low side"
        low_side_panel.title_align = "left"
        high_side_panel = display_diff_object(
            diff.high_state if diff.high_obj is not None else None
        )
        high_side_panel.title = "High side"
        high_side_panel.title_align = "left"

        grouped_panels = Group(low_side_panel, high_side_panel)

        diff_panel = Panel(
            grouped_panels,
            title=title,
            title_align="left",
            box=box.ROUNDED,
            expand=False,
            padding=(1, 2),
        )

        if level > 0:
            diff_panel = Padding(diff_panel, (0, 0, 0, 5 * level))

        console.print(diff_panel)
