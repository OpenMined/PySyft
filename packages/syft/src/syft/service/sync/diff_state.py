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
from ..code.user_code import UserCode
from ..job.job_stash import Job
from ..log.log import SyftLog
from ..project.project import Project
from ..request.request import Request
from ..request.request import UserCodeStatusChange
from ..response import SyftError
from .sync_state import SyncState

sketchy_tab = "‎ " * 4

only_attr_dict = {
    SyftLog.__name__: ["stdout", "stderr"],
    Job.__name__: [
        "result",
        "resolved",
        "status",
        "log_id",
        "parent_job_id",
        "n_iters",
        "current_iter",
        "creation_time",
        "updated_at",
        "user_code_id",
    ],
    UserCode.__name__: [
        "raw_code",
        "input_policy_type",
        "input_policy_init_kwargs",
        "input_policy_state",
        "output_policy_type",
        "output_policy_init_kwargs",
        "output_policy_state",
        "parsed_code",
        # 'service_func_name', 'unique_func_name', 'input_kwargs',
        # 'user_unique_func_name', 'code_hash', 'signature',
        "status",
        "submit_time",
        "nested_codes",
        # enclave_metadata, uses_domain, 'worker_pool_name',
    ],
    Request.__name__: [
        "requesting_user_name",
        "requesting_user_email",
        "requesting_user_institution",
        "request_time",
        "updated_at",
        "request_hash",
        "changes",
        "history",
    ],
    Project.__name__: None,
    ActionObject.__name__: None,
}


class DiffAttr(SyftObject):
    # version
    __canonical_name__ = "DiffAttr"
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
        return recursive_repr(self.low_attr)

    def __repr_high_side__(self):
        return recursive_repr(self.high_attr)


except_attrs_dict = {
    SyftLog.__name__: None,
    Job.__name__: None,
    UserCode.__name__: None,
    Request.__name__: None,
    Project.__name__: None,
    ActionObject.__name__: None,
}


class DiffList(DiffAttr):
    # version
    __canonical_name__ = "DiffList"
    __version__ = SYFT_OBJECT_VERSION_1
    diff_ids: List[int] = []
    new_low_ids: List[int] = []
    new_high_ids: List[int] = []


class DiffDict(DiffAttr):
    # version
    __canonical_name__ = "DiffDict"
    __version__ = SYFT_OBJECT_VERSION_1
    diff_keys: List[Any] = []
    new_low_keys: List[Any] = []
    new_high_keys: List[Any] = []


def check_linked_obj(low_linked_obj, high_linked_obj):
    check_id = low_linked_obj.id == high_linked_obj.id
    check_service_type = low_linked_obj.service_type == high_linked_obj.service_type
    check_object_type = low_linked_obj.object_type == high_linked_obj.object_type
    check_object_uid = low_linked_obj.object_uid == high_linked_obj.object_uid
    return check_id and check_service_type and check_object_type and check_object_uid


def check_change(low_change, high_change):
    check_value = low_change.value == high_change.value
    check_lo = check_linked_obj(low_change.linked_obj, high_change.linked_obj)
    check_ns = low_change.nested_solved == high_change.nested_solved
    check_mt = low_change.match_type == high_change.match_type
    return check_value and check_lo and check_ns and check_mt


def get_diff_dict(low_dict, high_dict, check_func=None):
    diff_keys = []

    low_dict_keys = set(low_dict.keys())
    high_dict_keys = set(high_dict.keys())

    common_keys = low_dict_keys & high_dict_keys
    new_low_keys = low_dict_keys - high_dict_keys
    new_high_keys = high_dict_keys - low_dict_keys

    for key in common_keys:
        if check_func:
            if not check_func(low_dict[key], high_dict[key]):
                diff_keys.append(key)
        elif low_dict[key] != high_dict[key]:
            diff_keys.append(key)

    return diff_keys, list(new_low_keys), list(new_high_keys)


def get_diff_list(low_list, high_list, check_func=None):
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
        if check_func:
            if not check_func(low_list[i], high_list[i]):
                diff_ids.append(i)
        elif low_list[i] != high_list[i]:
            diff_ids.append(i)

    return diff_ids, new_low_ids, new_high_ids


def get_diff_request(low_request, high_request):
    diff_attrs = []

    # Sanity check
    if low_request.id != high_request.id:
        raise Exception("Not the same id for low side and high side requests")

    basic_attrs = [
        "requesting_user_name",
        "requesting_user_email",
        "requesting_user_institution",
        "approving_user_verify_key",
        "request_time",
        "updated_at",
        "request_hash",
    ]

    for attr in basic_attrs:
        low_attr = getattr(low_request, attr)
        high_attr = getattr(high_request, attr)
        if low_attr != high_attr:
            diff_attr = DiffAttr(attr_name=attr, low_attr=low_attr, high_attr=high_attr)
            diff_attrs.append(diff_attr)

    change_diffs = get_diff_list(
        low_request.changes, high_request.changes, check_func=check_change
    )
    for d in change_diffs:
        if len(d) != 0:
            change_diff = DiffList(
                attr_name="change",
                low_attr=low_request.changes,
                high_attr=high_request.changes,
                diff_ids=change_diffs[0],
                new_low_ids=change_diffs[1],
                new_high_ids=change_diffs[2],
            )
            diff_attrs.append(change_diff)
            break

    # history_diffs = get_diff_list(low_request.history, high_request.history)
    # for l in history_diffs:
    #     if len(l) != 0:
    #         history_diff = DiffList(
    #             attr_name="history",
    #             low_attr=low_request.history,
    #             high_attr=high_request.history,
    #             diff_ids=history_diffs[0],
    #             new_low_ids=history_diffs[1],
    #             new_high_ids=history_diffs[2]
    #         )
    #         diff_attrs.append(history_diff)
    #         break

    return diff_attrs


def get_diff_user_code(low_code, high_code):
    diff_attrs = []

    # Sanity check
    if low_code.id != high_code.id:
        raise Exception("Not the same id for low side and high side requests")

    # Non basic attrs:
    # "nested_codes",
    # "status"

    basic_attrs = [
        "raw_code",
        "input_policy_type",
        "input_policy_init_kwargs",
        "input_policy_state",
        "output_policy_type",
        "output_policy_init_kwargs",
        "output_policy_state",
        "parsed_code",
        "service_func_name",
        "unique_func_name",
        "user_unique_func_name",
        "code_hash",
        "signature",
        "input_kwargs",
        # "enclave_metadata",
        "submit_time",
        "uses_domain",
        "worker_pool_name",
    ]

    for attr in basic_attrs:
        low_attr = getattr(low_code, attr)
        high_attr = getattr(high_code, attr)
        if low_attr != high_attr:
            diff_attr = DiffAttr(attr_name=attr, low_attr=low_attr, high_attr=high_attr)
            diff_attrs.append(diff_attr)

    low_status = list(low_code.status.status_dict.values())[0]
    high_status = list(high_code.status.status_dict.values())[0]

    if low_status != high_status:
        diff_attr = DiffAttr(
            attr_name="status", low_attr=low_status, high_attr=high_status
        )
        diff_attrs.append(diff_attr)

    return diff_attrs


def get_diff_log(low_log, high_log):
    diff_attrs = []

    # Sanity check
    if low_log.id != high_log.id:
        raise Exception("Not the same id for low side and high side requests")

    basic_attrs = ["stdout", "stderr"]

    for attr in basic_attrs:
        low_attr = getattr(low_log, attr)
        high_attr = getattr(high_log, attr)
        if low_attr != high_attr:
            diff_attr = DiffAttr(attr_name=attr, low_attr=low_attr, high_attr=high_attr)
            diff_attrs.append(diff_attr)

    return diff_attrs


def get_diff_job(low_job, high_job):
    diff_attrs = []

    # Sanity check
    if low_job.id != high_job.id:
        raise Exception("Not the same id for low side and high side requests")

    # Non basic attrs:
    # "nested_codes",
    # "status"

    basic_attrs = [
        "result",
        "resolved",
        "status",
        "log_id",
        "parent_job_id",
        "n_iters",
        "current_iter",
        "creation_time",
        "job_pid",
        "job_worker_id",
        "updated_at",
        "user_code_id",
    ]

    for attr in basic_attrs:
        low_attr = getattr(low_job, attr)
        high_attr = getattr(high_job, attr)
        if low_attr != high_attr:
            diff_attr = DiffAttr(attr_name=attr, low_attr=low_attr, high_attr=high_attr)
            diff_attrs.append(diff_attr)

    return diff_attrs


def get_diff_action_object(low_action_object, high_action_object):
    diff_attrs = []

    # Sanity check
    if low_action_object.id != high_action_object.id:
        raise Exception("Not the same id for low side and high side requests")

    # Non basic attrs:
    # "nested_codes",
    # "status"

    # basic_attrs = [
    #     "__attr_searchable__",
    #     "syft_action_data_cache",
    #     "syft_blob_storage_entry_id",
    #     "syft_pointer_type",
    #     "syft_parent_hashes",
    #     "syft_parent_op",
    #     "syft_parent_args",
    #     "syft_parent_kwargs",
    #     "syft_history_hash",
    #     "syft_internal_type",
    #     "syft_node_uid",
    #     "_syft_pre_hooks__",
    #     "_syft_post_hooks__",
    #     "syft_twin_type",
    #     "syft_action_data_type",
    #     "syft_action_data_repr_",
    #     "syft_action_data_str_",
    #     "syft_has_bool_attr",
    #     "syft_resolve_data",
    #     "syft_created_at",
    #     "syft_resolved",
    # ]

    # for attr in basic_attrs:
    #     low_attr = getattr(low_action_object, attr)
    #     high_attr = getattr(high_action_object, attr)
    #     if low_attr != high_attr:
    #         diff_attr = DiffAttr(attr_name=attr, low_attr=low_attr, high_attr=high_attr)
    #         diff_attrs.append(diff_attr)
    low_data = low_action_object.syft_action_data
    high_data = high_action_object.syft_action_data
    if low_data != high_data:
        diff_attr = DiffAttr(
            attr_name="syft_action_data", low_attr=low_data, high_attr=high_data
        )
        diff_attrs.append(diff_attr)
    return diff_attrs


func_dict = {  #
    SyftLog.__name__: get_diff_log,
    Job.__name__: get_diff_job,
    UserCode.__name__: get_diff_user_code,
    Request.__name__: get_diff_request,
    Project.__name__: None,
    ActionObject.__name__: get_diff_action_object,
    # UserCodeStatusChange.__name__: None,
}


def check_diff(low_obj, high_obj, obj_type):
    attrs_to_check = only_attr_dict[obj_type]
    for attr in attrs_to_check:
        low_attr = getattr(low_obj, attr)
        high_attr = getattr(high_obj, attr)

        # if isinstance(low_attr, SyftObject) and isinstance(high_attr, SyftObject)

        if low_attr != high_attr:
            return False

    return True


# relative
# check_dict_func = {
#     SyftLogV2.__name__: check_log,
#     Job.__name__: check_job,
#     UserCode.__name__: check_code,
#     Request.__name__: check_request,
#     Project.__name__: check_project
# }


class SyftString(SyftObject):
    # version
    __canonical_name__ = "SyftString"
    __version__ = SYFT_OBJECT_VERSION_1
    string: str


def recursive_repr(value_attr, no_tabs=0):
    new_no_tabs = no_tabs + 1 if no_tabs != 0 else 2
    if isinstance(value_attr, List):
        list_repr = "[\n"
        for elem in value_attr:
            list_repr += recursive_repr(elem, no_tabs=new_no_tabs) + "\n"
        list_repr += f"{sketchy_tab * (new_no_tabs-1)}]"
        return list_repr
    elif isinstance(value_attr, Dict):
        dict_repr = "[\n"
        for key, elem in value_attr.items:
            elem_repr = {recursive_repr(elem, no_tabs=new_no_tabs)}
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


class Diff(SyftObject):  # StateTuple (compare 2 objects)
    # version
    __canonical_name__ = "Diff"
    __version__ = SYFT_OBJECT_VERSION_1
    low_obj: Optional[SyftObject] = None
    high_obj: Optional[SyftObject] = None
    obj_type: Type
    merge_state: str
    diff_list: List[DiffAttr] = []

    __repr_attrs__ = [
        # "object_type",
        "merge_state",
        "low_state",
        "high_state",
    ]

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
            return "n/a"
        if isinstance(self.low_obj, ActionObject):
            return self.low_obj.__repr__()
        if self.high_obj is None:
            attrs_str = ""
            attrs = getattr(self.low_obj, "__repr_attrs__", [])
            for attr in attrs:
                value = getattr(self.low_obj, attr)
                attrs_str += f"{sketchy_tab}{attr} = {recursive_repr(value)}\n"
            attrs_str = attrs_str[:-1]
            return f"class {self.object_type}:\n{attrs_str}"
        attr_text = f"class {self.object_type}:\n"
        for diff in self.diff_list:
            attr_text += (
                f"{sketchy_tab}{diff.attr_name}={diff.__repr_low_side__()}," + "\n"
            )

        if len(self.diff_list) > 0:
            attr_text = attr_text[:-2]

        return attr_text
        # return "DIFF"

    @property
    def high_state(self):
        if self.high_obj is None:
            return "n/a"

        if isinstance(self.high_obj, ActionObject):
            return self.high_obj.__repr__()

        if self.low_obj is None:
            attrs_str = ""
            attrs = getattr(self.high_obj, "__repr_attrs__", [])
            for attr in attrs:
                value = getattr(self.high_obj, attr)
                attrs_str += f"{sketchy_tab}{attr} = {recursive_repr(value)}\n"
            attrs_str = attrs_str[:-1]
            return f"class {self.object_type}:\n{attrs_str}"
        attr_text = f"class {self.object_type}:\n"
        for diff in self.diff_list:
            attr_text += (
                f"{sketchy_tab}{diff.attr_name}={diff.__repr_high_side__()}," + "\n"
            )
        if len(self.diff_list) > 0:
            attr_text = attr_text[:-2]

        return attr_text
        # return "DIFF"

    def get_obj(self):
        if self.merge_state == "NEW":
            return self.low_obj if self.low_obj is not None else self.high_obj
        else:
            return "Error"

    def _repr_html_(self) -> Any:
        if self.low_obj is None and self.high_obj is None:
            return SyftError(message="Something broke")

        if self.low_obj is None:
            if hasattr(self.high_obj, "_repr_html_"):
                obj_repr = self.high_obj._repr_html_()
            elif hasattr(self.high_obj, "_inner_repr"):
                obj_repr = self.high_obj._inner_repr()
            else:
                obj_repr = self.__repr__()
            return (
                f"""
    <style>
    {fonts_css}
    .syft-dataset {{color: {SURFACE[options.color_theme]};}}
    .syft-dataset h3,
    .syft-dataset p
        {{font-family: 'Open Sans';}}
        {ITABLES_CSS}
    </style>
    <div class='syft-diff'>
    <h3>{self.object_type} Diff (New {self.object_type}  on the High Side):</h3>
    """
                + obj_repr
            )

        if self.high_obj is None:
            if hasattr(self.low_obj, "_repr_html_"):
                obj_repr = self.low_obj._repr_html_()
            elif hasattr(self.low_obj, "_inner_repr"):
                obj_repr = self.low_obj._inner_repr()
            else:
                obj_repr = self.__repr__()
            return (
                f"""
    <style>
    {fonts_css}
    .syft-dataset {{color: {SURFACE[options.color_theme]};}}
    .syft-dataset h3,
    .syft-dataset p
        {{font-family: 'Open Sans';}}
        {ITABLES_CSS}
    </style>
    <div class='syft-diff'>
    <h3>{self.object_type} Diff (New {self.object_type}  on the Low Side):</h3>
    """
                + obj_repr
            )

        if self.merge_state == "SAME":
            attr_text = "No changes between low side and high side"
        else:
            attr_text = ""
            for diff in self.diff_list:
                attr_text += diff.__repr__() + "<br>"

            attr_text = attr_text.replace("\n", "<br>")
            # print("New lines", res)

        return f"""
        <style>
        {fonts_css}
        .syft-dataset {{color: {SURFACE[options.color_theme]};}}
        .syft-dataset h3,
        .syft-dataset p
            {{font-family: 'Open Sans';}}
            {ITABLES_CSS}
        </style>
        <div class='syft-diff'>
        <h3>{self.object_type} Diff</h3>
        {attr_text}
        """


def get_type(obj):
    if isinstance(obj, Project):
        return Project
    if isinstance(obj, Request):
        return Request
    if isinstance(obj, UserCode):
        return UserCode
    if isinstance(obj, Job):
        return Job
    if isinstance(obj, SyftLog):
        return SyftLog
    if isinstance(obj, ActionObject):
        return ActionObject
    raise NotImplementedError


def get_merge_state(low_obj, high_obj, obj_type):
    if low_obj is None and high_obj is None:
        return "Error"
    if low_obj is None or high_obj is None:
        return "NEW"
    # if check_diff(low_obj, high_obj, obj_type):
    #     return "SAME"
    override_func = func_dict.get(obj_type.__name__, None)
    if override_func:
        diffs = override_func(low_obj, high_obj)
        if len(diffs) == 0:
            return "SAME"

    return "DIFF"


class DiffState(SyftObject):
    __canonical_name__ = "DiffState"
    __version__ = SYFT_OBJECT_VERSION_1

    obj_uid_to_diff: Dict[UID, Diff] = {}
    dependencies: Dict[UID, List[UID]] = {}

    @classmethod
    def from_sync_state(
        cls: Type["DiffState"], low_state: SyncState, high_state: SyncState
    ) -> "DiffState":
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
    def diffs(self) -> List[Diff]:
        # Returns a list of diffs, in depth-first order.
        return [diff for hierarchy in self.hierarchies for diff, _ in hierarchy]

    @property
    def hierarchies(self) -> List[List[Tuple[Diff, int]]]:
        # Returns a list of hierarchies, where each hierarchy is a list of tuples (Diff, level),
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
        obj_type = get_type(low_obj if low_obj is not None else high_obj)

        if low_obj is None or high_obj is None:
            diff = Diff(
                low_obj=low_obj,
                high_obj=high_obj,
                obj_type=obj_type,
                merge_state="NEW",
                diff_list=[],
            )
        else:
            override_func = func_dict.get(obj_type.__name__, None)
            diff_list = override_func(low_obj, high_obj)
            if len(diff_list) == 0:
                merge_state = "SAME"
            else:
                merge_state = "DIFF"
            diff = Diff(
                low_obj=low_obj,
                high_obj=high_obj,
                obj_type=obj_type,
                merge_state=merge_state,
                diff_list=diff_list,
            )
        self.obj_uid_to_diff[diff.object_id] = diff

    def objs_to_sync(self):
        objs = []
        for diff in self.diffs:
            if diff.merge_state == "NEW":
                objs.append(diff.get_obj())
        return objs


class ResolveState(SyftObject):
    __canonical_name__ = "SyncUpdate"
    __version__ = SYFT_OBJECT_VERSION_1

    create_objs: List[SyftObject] = []
    update_objs: List[SyftObject] = []
    delete_objs: List[SyftObject] = []

    def add(self, new_state: "ResolveState") -> None:
        self.create_objs.extend(new_state.create_objs)
        self.update_objs.extend(new_state.update_objs)
        self.delete_objs.extend(new_state.delete_objs)

    def __repr__(self):
        return (
            f"ResolveState(\n"
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


def display_diff_hierarchy(diff_hierarchy: List[Tuple[Diff, int]]):
    console = Console()

    for diff, level in diff_hierarchy:
        title = (
            f"{diff.obj_type.__name__}({diff.object_id}) - State: {diff.merge_state}"
        )

        low_side_panel = display_diff_object(diff.low_state if diff.low_obj else None)
        low_side_panel.title = "Low side"
        low_side_panel.title_align = "left"
        high_side_panel = display_diff_object(
            diff.high_state if diff.high_obj else None
        )
        high_side_panel.title = "High side"
        high_side_panel.title_align = "left"

        grouped_panels = Group(low_side_panel, high_side_panel)

        diff_panel = Panel(
            grouped_panels,
            title=title,
            title_align="left",
            box=box.HEAVY_EDGE,
            expand=False,
            padding=(1, 2),
        )

        if level > 0:
            diff_panel = Padding(diff_panel, (0, 0, 0, 5 * level))
        # Printing the main Panel using Rich Console
        console.print(diff_panel)


def resolve_diff(diff: Diff, decision: str) -> ResolveState:
    resolved_diff_low = ResolveState()
    resolved_diff_high = ResolveState()

    # No diff, return empty resolvestate
    if diff.merge_state == "SAME":
        return resolved_diff_low, resolved_diff_high

    elif diff.merge_state == "NEW":
        low_is_none = diff.low_obj is None
        high_is_none = diff.high_obj is None
        if low_is_none and high_is_none:
            raise ValueError(
                f"Diff {diff.id} is missing objects: both low and high objects are None"
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
