"""
 Things to figure out:
    * create a better test scenario with multiple different stages
    * recursively add subobject in sync_state
    * efficiently check if object was already included in a previous upper level object
    * create state datastructure for better UX
    * check list for each type of object to see if diff must be included or not
"""
from typing import Any, Dict, List, Optional, Type
from IPython.core.display import display, HTML, Markdown

from syft.service.action.action_data_empty import ActionDataEmpty
from ..types.syft_object import SYFT_OBJECT_VERSION_1, SyftObject
from ..service.code.user_code import UserCode

from ..service.project.project import Project
from ..service.request.request import Request
from ..service.job.job_stash import Job
from ..service.log.log import SyftLog
from ..util.fonts import ITABLES_CSS, fonts_css
from ..util.colors import SURFACE
from ..util import options
from ..service.action.action_object import ActionObject
from syft import UID
from typing import Set

"""
How to check differences between two objects:
    * by default merge every attr
    * check if there is a custom implementation of the check function
    * check if there are exceptions we do not want to merge
    * check if there are some restrictions on the attr set
"""

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
        return f"{self.low_attr}"

    def __repr_high_side__(self):
        return f"{self.high_attr}"


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
    for l in change_diffs:
        if len(l) != 0:
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


# check_dict_func = {
#     SyftLogV2.__name__: check_log,
#     Job.__name__: check_job,
#     UserCode.__name__: check_code,
#     Request.__name__: check_request,
#     Project.__name__: check_project
# }
from syft.service.response import SyftError


class SyftString(SyftObject):
    # version
    __canonical_name__ = "SyftString"
    __version__ = SYFT_OBJECT_VERSION_1
    string: str


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
    def object_type(self):
        return self.obj_type.__name__

    @property
    def low_state(self):
        if self.low_obj is None:
            return "n/a"
        if self.high_obj is None:
            return "NEW"
        attr_text = f"{self.object_type}("
        for diff in self.diff_list:
            attr_text += f"{diff.attr_name}={diff.__repr_low_side__()}," + "\n"

        if len(self.diff_list) > 0:
            attr_text = attr_text[:-2] + ")"
        else:
            attr_text += ")"
        return attr_text
        # return "DIFF"

    @property
    def high_state(self):
        if self.high_obj is None:
            return "n/a"

        if self.low_obj is None:
            return "NEW"
        attr_text = f"{self.object_type}("
        for diff in self.diff_list:
            attr_text += f"{diff.attr_name}={diff.__repr_high_side__()}," + "\n"
        if len(self.diff_list) > 0:
            attr_text = attr_text[:-2] + ")"
        else:
            attr_text += ")"
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

            import re

            res = [i.start() for i in re.finditer("\t", attr_text)]

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
    raise NotImplemented


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
    diffs: List[Diff] = []

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
        self.diffs.append(diff)

    def objs_to_sync(self):
        objs = []
        for diff in self.diffs:
            if diff.merge_state == "NEW":
                objs.append(diff.get_obj())
        return objs


class SyncState(SyftObject):
    __canonical_name__ = "SyncState"
    __version__ = SYFT_OBJECT_VERSION_1

    objs_by_type: Dict[str, Dict[UID, SyftObject]] = {}

    # TODO Full list of dependencies for each object (recursively)
    dependencies: Dict[UID, Set[UID]] = {}

    def __getitem__(self, key: UID) -> Any:
        for items in self.objs_by_type.values():
            if key in items:
                return items[key]
        raise KeyError(f"Object with UID {key} not found in SyncState")

    @property
    def all_ids(self) -> Set[UID]:
        ids = set()
        for obj_dict in self.objs_by_type.values():
            ids.update(obj_dict.keys())
        return ids

    def _build_dependencies(self) -> None:
        all_ids = self.all_ids
        # TODO rewrite recursion, dont need to go through the whole object tree for each object
        for obj_type in self.objs_by_type:
            for obj_id, obj in self.objs_by_type[obj_type].items():
                if hasattr(obj, "get_dependencies"):
                    dependencies = obj.get_dependencies()[obj_id]
                    self.dependencies[obj_id] = {
                        d.id for d in dependencies if d.id in all_ids
                    }
                else:
                    self.dependencies[obj_id] = set()

    def get_hierarchical_order(self) -> List[Tuple[SyftObject, int]]:
        raise NotImplementedError()


def get_sync_state(self):
    sync_state = SyncState()

    projects = self.api.services.project.get_all()
    sync_state.objs_by_type["Project"] = {p.id: p for p in projects}

    requests = self.api.services.request.get_all()
    sync_state.objs_by_type["Request"] = {r.id: r for r in requests}

    user_codes = self.api.services.code.get_all()
    sync_state.objs_by_type["UserCode"] = {c.id: c for c in user_codes}

    jobs = self.api.services.job.get_all()
    sync_state.objs_by_type["Job"] = {j.id: j for j in jobs}

    logs = self.api.services.log.get_all()
    sync_state.objs_by_type["SyftLog"] = {l.id: l for l in logs}

    # TODO workaround, we only need action objects from output policies for now
    action_objects = []
    for code in user_codes:
        action_objects.extend(code.get_all_output_action_objects())
    sync_state.objs_by_type["ActionObject"] = {a.id: a for a in action_objects}

    sync_state._build_dependencies()
    return sync_state


def compare_states(low_state, high_state) -> DiffState:
    diff_state = DiffState()

    for obj_type in low_state.objs_by_type.keys():
        low_objects = low_state.objs_by_type[obj_type]
        high_objects = high_state.objs_by_type[obj_type]
        all_ids = set(low_objects.keys()) | set(high_objects.keys())

        for obj_id in all_ids:
            low_obj = low_objects.get(obj_id, None)
            high_obj = high_objects.get(obj_id, None)
            diff_state.add_obj(low_obj, high_obj)

    return diff_state


def resolve(state: DiffState):
    low_new_objs = []
    high_new_objs = []
    # new_objs = state.objs_to_sync()
    for new_obj in state.diff_to_sync:
        if new_obj.merge_state == "NEW":
            if new_obj.low_obj is None:
                state_list = low_new_objs
                source = "LOW"
                destination = "HIGH"
                obj_to_sync = new_obj.high_obj
            if new_obj.high_obj is None:
                state_list = high_new_objs
                source = "HIGH"
                destination = "LOW"
                obj_to_sync = new_obj.low_obj
            if hasattr(obj_to_sync, "_repr_markdown_"):
                display(Markdown(obj_to_sync._repr_markdown_()))
            else:
                display(obj_to_sync)
            print(
                f"Do you approve moving this object from the {source} side to the {destination} side (approve/deny): ",
                flush=True,
            )
            while True:
                decision = input()
                if decision == "approve":
                    state_list.append(obj_to_sync)
                    break
                elif decision == "deny":
                    break
                else:
                    print("Please write `approve` or `deny`:", flush=True)
        if new_obj.merge_state == "DIFF":
            pass

    return low_new_objs, high_new_objs
