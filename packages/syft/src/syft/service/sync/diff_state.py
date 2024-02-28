"""
How to check differences between two objects:
    * by default merge every attr
    * check if there is a custom implementation of the check function
    * check if there are exceptions we do not want to merge
    * check if there are some restrictions on the attr set
"""

# stdlib
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type

# third party
from rich import box
from rich.console import Console
from rich.console import Group
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel

# relative
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ...types.uid import LineageID
from ...types.uid import UID
from ...util import options
from ...util.colors import SURFACE
from ...util.fonts import ITABLES_CSS
from ...util.fonts import fonts_css
from ..action.action_object import ActionObject
from ..request.request import UserCodeStatusChange
from ..response import SyftError
from .sync_state import SyncState

sketchy_tab = "‎ " * 4


class AttrDiff(SyftObject):
    # version
    __canonical_name__ = "AttrDiff"
    __version__ = SYFT_OBJECT_VERSION_1
    attr_name: str
    low_attr: Any
    high_attr: Any

    def __repr__(self):
        return f"""{self.attr_name}:
    Low Side value: {self.low_attr}
    High Side value: {self.high_attr}
    """

    def __repr_low_side__(self):
        return recursive_attr_repr(self.low_attr)

    def __repr_high_side__(self):
        return recursive_attr_repr(self.high_attr)


class ListDiff(AttrDiff):
    # version
    __canonical_name__ = "ListDiff"
    __version__ = SYFT_OBJECT_VERSION_1
    diff_ids: List[int] = []
    new_low_ids: List[int] = []
    new_high_ids: List[int] = []

    @property
    def is_empty(self):
        return (
            len(self.diff_ids) == 0
            and len(self.new_low_ids) == 0
            and len(self.new_high_ids) == 0
        )

    @classmethod
    def from_lists(cls, attr_name, low_list, high_list):
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
            # if hasattr(low_list[i], 'syft_eq'):
            #     if not low_list[i].syft_eq(high_list[i]):
            #         diff_ids.append(i)
            if low_list[i] != high_list[i]:
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


# class DictDiff(AttrDiff):
#     # version
#     __canonical_name__ = "DictDiff"
#     __version__ = SYFT_OBJECT_VERSION_1
#     diff_keys: List[Any] = []
#     new_low_keys: List[Any] = []
#     new_high_keys: List[Any] = []

# def get_diff_dict(low_dict, high_dict, check_func=None):
#     diff_keys = []

#     low_dict_keys = set(low_dict.keys())
#     high_dict_keys = set(high_dict.keys())

#     common_keys = low_dict_keys & high_dict_keys
#     new_low_keys = low_dict_keys - high_dict_keys
#     new_high_keys = high_dict_keys - low_dict_keys

#     for key in common_keys:
#         if check_func:
#             if not check_func(low_dict[key], high_dict[key]):
#                 diff_keys.append(key)
#         elif low_dict[key] != high_dict[key]:
#             diff_keys.append(key)

#     return diff_keys, list(new_low_keys), list(new_high_keys)


def recursive_attr_repr(value_attr, no_tabs=0):
    new_no_tabs = no_tabs + 1 if no_tabs != 0 else 2
    if isinstance(value_attr, List):
        list_repr = "[\n"
        for elem in value_attr:
            list_repr += recursive_attr_repr(elem, no_tabs=new_no_tabs) + "\n"
        list_repr += f"{sketchy_tab * (new_no_tabs-1)}]"
        return list_repr
    elif isinstance(value_attr, Dict):
        dict_repr = "[\n"
        for key, elem in value_attr.items:
            elem_repr = {recursive_attr_repr(elem, no_tabs=new_no_tabs)}
            dict_repr += f"{sketchy_tab * no_tabs}{key}: {elem_repr}\n"
        dict_repr += f"{sketchy_tab * (new_no_tabs-1)}]"
        return
    elif isinstance(value_attr, UserCodeStatusChange):
        return f"{sketchy_tab*no_tabs}UserCodeStatusChange"
    # elif isinstance(value_attr, str):
    #     if len(value_attr.split(",")) > 1:
    #         repr_str = ""
    #         for sub_string in value_attr.split(","):
    #             repr_str += f"{sketchy_tab*(new_no_tabs-1)}{sub_string},\n"
    #         return repr_str[:-1]

    elif isinstance(value_attr, bytes):
        value_attr = repr(value_attr)
        if len(value_attr) > 50:
            value_attr = value_attr[:50] + "..."
    return f"{sketchy_tab*no_tabs}{value_attr}"


class ObjectDiff(SyftObject):  # StateTuple (compare 2 objects)
    # version
    __canonical_name__ = "ObjectDiff"
    __version__ = SYFT_OBJECT_VERSION_1
    low_obj: Optional[SyftObject] = None
    high_obj: Optional[SyftObject] = None
    obj_type: Type
    diff_list: List[AttrDiff] = []

    __repr_attrs__ = [
        # "object_type",
        # "merge_state",
        "low_state",
        "high_state",
    ]

    @property
    def merge_state(self):
        if self.low_obj is None or self.high_obj is None:
            return "NEW"

        if len(self.diff_list) == 0:
            return "SAME"

        return "DIFF"

    @property
    def object_id(self):
        uid = self.low_obj.id if self.low_obj is not None else self.high_obj.id
        if isinstance(uid, LineageID):
            return uid.id
        return uid

    @property
    def object_type(self):
        return self.obj_type.__name__

    @property
    def low_state(self):
        if self.low_obj is None:
            return "-"
        if isinstance(self.low_obj, ActionObject):
            return self.low_obj.__repr__()
        if self.high_obj is None:
            attrs_str = ""
            attrs = getattr(self.low_obj, "__repr_attrs__", [])
            for attr in attrs:
                value = getattr(self.low_obj, attr)
                attrs_str += f"{sketchy_tab}{attr} = {recursive_attr_repr(value)}\n"
            attrs_str = attrs_str[:-1]
            return f"NEW\n\nclass {self.object_type}:\n{attrs_str}"
        attr_text = f"class {self.object_type}:\n"
        for diff in self.diff_list:
            attr_text += (
                f"{sketchy_tab}{diff.attr_name}={diff.__repr_low_side__()}," + "\n"
            )

        if len(self.diff_list) > 0:
            attr_text = attr_text[:-2]

        return attr_text

    @property
    def high_state(self):
        if self.high_obj is None:
            return "-"

        if isinstance(self.high_obj, ActionObject):
            return self.high_obj.__repr__()

        if self.low_obj is None:
            attrs_str = ""
            attrs = getattr(self.high_obj, "__repr_attrs__", [])
            for attr in attrs:
                value = getattr(self.high_obj, attr)
                attrs_str += f"{sketchy_tab}{attr} = {recursive_attr_repr(value)}\n"
            attrs_str = attrs_str[:-1]
            return f"NEW\n\nclass {self.object_type}:\n{attrs_str}"
        attr_text = f"class {self.object_type}:\n"
        for diff in self.diff_list:
            attr_text += (
                f"{sketchy_tab}{diff.attr_name}={diff.__repr_high_side__()}," + "\n"
            )
        if len(self.diff_list) > 0:
            attr_text = attr_text[:-2]

        return attr_text

    def get_obj(self):
        if self.merge_state == "NEW":
            return self.low_obj if self.low_obj is not None else self.high_obj
        else:
            return "Error"

    def _repr_html_(self) -> str:
        if self.low_obj is None and self.high_obj is None:
            return SyftError(message="Something broke")

        base_str = f"""
        <style>
        {fonts_css}
        .syft-dataset {{color: {SURFACE[options.color_theme]};}}
        .syft-dataset h3,
        .syft-dataset p
            {{font-family: 'Open Sans';}}
            {ITABLES_CSS}
        </style>
        <div class='syft-diff'>
        """

        if self.low_obj is None:
            if hasattr(self.high_obj, "_repr_html_"):
                obj_repr = self.high_obj._repr_html_()
            elif hasattr(self.high_obj, "_inner_repr"):
                obj_repr = self.high_obj._inner_repr()
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
                obj_repr = self.low_obj._repr_html_()
            elif hasattr(self.low_obj, "_inner_repr"):
                obj_repr = self.low_obj._inner_repr()
            else:
                obj_repr = self.__repr__()
            attr_text = (
                f"""
    <h3>{self.object_type} ObjectDiff (New {self.object_type}  on the High Side):</h3>
    """
                + obj_repr
            )

        elif self.merge_state == "SAME":
            obj_repr = "No changes between low side and high side"
        else:
            obj_repr = ""
            for diff in self.diff_list:
                obj_repr += diff.__repr__() + "<br>"

            obj_repr = obj_repr.replace("\n", "<br>")
            # print("New lines", res)

        attr_text = f"<h3>{self.object_type} ObjectDiff:</h3>\n{obj_repr}"
        return base_str + attr_text


class NodeDiff(SyftObject):
    __canonical_name__ = "NodeDiff"
    __version__ = SYFT_OBJECT_VERSION_1

    obj_uid_to_diff: Dict[UID, ObjectDiff] = {}
    dependencies: Dict[UID, List[UID]] = {}

    @classmethod
    def from_sync_state(
        cls: Type["NodeDiff"], low_state: SyncState, high_state: SyncState
    ) -> "NodeDiff":
        diff_state = cls()

        all_ids = set(low_state.objects.keys()) | set(high_state.objects.keys())

        for obj_id in all_ids:
            low_obj = low_state.objects.get(obj_id, None)
            high_obj = high_state.objects.get(obj_id, None)
            diff_state.add_obj(low_obj, high_obj)

        diff_state._init_dependencies(low_state, high_state)
        return diff_state

    def _init_dependencies(self, low_state: SyncState, high_state: SyncState) -> None:
        all_parents = set(low_state.dependencies.keys()) | set(
            high_state.dependencies.keys()
        )
        for parent in all_parents:
            low_deps = low_state.dependencies.get(parent, [])
            high_deps = high_state.dependencies.get(parent, [])
            self.dependencies[parent] = list(set(low_deps) | set(high_deps))

    @property
    def diffs(self) -> List[ObjectDiff]:
        # Returns a list of diffs, in depth-first order.
        return [diff for hierarchy in self.hierarchies for diff, _ in hierarchy]

    def _repr_html_(self) -> Any:
        return self.diffs._repr_html_()

    @property
    def hierarchies(self) -> List[List[Tuple[ObjectDiff, int]]]:
        # Returns a list of hierarchies, where each hierarchy is a list of tuples (ObjectDiff, level),
        # in depth-first order.

        # Each hierarchy only contains one root, at the first position
        # Example: [(Diff1, 0), (Diff2, 1), (Diff3, 2), (Diff4, 1)]
        # Diff1
        # -- Diff2
        # ---- Diff3
        # -- Diff4

        def _build_hierarchy_helper(uid, level=0):
            result = [(uid, level)]
            if uid in self.dependencies:
                for child_uid in self.dependencies[uid]:
                    result.extend(_build_hierarchy_helper(child_uid, level + 1))
            return result

        hierarchies = []
        all_ids = set(self.obj_uid_to_diff.keys())
        child_ids = {child for deps in self.dependencies.values() for child in deps}
        # Root ids are object ids with no parents
        root_ids = list(all_ids - child_ids)

        for root_uid in root_ids:
            uid_hierarchy = _build_hierarchy_helper(root_uid)
            diff_hierarchy = [
                (self.obj_uid_to_diff[uid], level) for uid, level in uid_hierarchy
            ]
            hierarchies.append(diff_hierarchy)

        return hierarchies

    def add_obj(self, low_obj, high_obj):
        if low_obj is None and high_obj is None:
            raise Exception("Both objects are None")
        obj_type = type(low_obj if low_obj is not None else high_obj)

        if low_obj is None or high_obj is None:
            diff = ObjectDiff(
                low_obj=low_obj,
                high_obj=high_obj,
                obj_type=obj_type,
                diff_list=[],
            )
        else:
            diff_list = high_obj.get_diffs(low_obj)
            diff = ObjectDiff(
                low_obj=low_obj,
                high_obj=high_obj,
                obj_type=obj_type,
                diff_list=diff_list,
            )
        self.obj_uid_to_diff[diff.object_id] = diff

    def objs_to_sync(self):
        objs = []
        for diff in self.diffs:
            if diff.merge_state == "NEW":
                objs.append(diff.get_obj())
        return objs


class ResolvedSyncState(SyftObject):
    __canonical_name__ = "SyncUpdate"
    __version__ = SYFT_OBJECT_VERSION_1

    create_objs: List[SyftObject] = []
    update_objs: List[SyftObject] = []
    delete_objs: List[SyftObject] = []

    def add(self, new_state: "ResolvedSyncState") -> None:
        self.create_objs.extend(new_state.create_objs)
        self.update_objs.extend(new_state.update_objs)
        self.delete_objs.extend(new_state.delete_objs)

    def __repr__(self):
        return (
            f"ResolvedSyncState(\n"
            f"  create_objs={self.create_objs},\n"
            f"  update_objs={self.update_objs},\n"
            f"  delete_objs={self.delete_objs}\n"
            f")"
        )


def display_diff_object(obj_state: Optional[str]) -> Panel:
    if obj_state is None:
        return Panel(Markdown("None"), box=box.ROUNDED, expand=False)
    return Panel(
        Markdown(f"```python\n{obj_state}\n```", code_theme="default"),
        box=box.ROUNDED,
        expand=False,
    )


def display_diff_hierarchy(diff_hierarchy: List[Tuple[ObjectDiff, int]]):
    console = Console()

    for diff, level in diff_hierarchy:
        title = (
            f"{diff.obj_type.__name__}({diff.object_id}) - State: {diff.merge_state}"
        )

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


def resolve_diff(diff: ObjectDiff, decision: str) -> ResolvedSyncState:
    resolved_diff_low = ResolvedSyncState()
    resolved_diff_high = ResolvedSyncState()

    # No diff, return empty resolved state
    if diff.merge_state == "SAME":
        return resolved_diff_low, resolved_diff_high

    elif diff.merge_state == "NEW":
        low_is_none = diff.low_obj is None
        high_is_none = diff.high_obj is None
        if low_is_none and high_is_none:
            raise ValueError(
                f"ObjectDiff {diff.id} is missing objects: both low and high objects are None"
            )
        if decision == "low" and high_is_none:
            resolved_diff_high.create_objs.append(diff.low_obj)
        elif decision == "low" and low_is_none:
            resolved_diff_high.delete_objs.append(diff.high_obj)
        elif decision == "high" and low_is_none:
            resolved_diff_low.create_objs.append(diff.high_obj)
        elif decision == "high" and high_is_none:
            resolved_diff_low.delete_objs.append(diff.low_obj)

    elif diff.merge_state == "DIFF":
        if decision == "low":
            resolved_diff_high.update_objs.append(diff.low_obj)
        else:  # decision == high
            resolved_diff_low.update_objs.append(diff.high_obj)

    return resolved_diff_low, resolved_diff_high
