# stdlib
import json
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

# relative
from ...client.api import APIRegistry
from ...client.enclave_client import EnclaveMetadata
from ...serde.serializable import serializable
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ...types.syft_object import SyftVerifyKey
from ...types.syft_object import get_repr_values_table
from ...types.uid import UID
from ...util.notebook_ui.notebook_addons import create_table_template
from ..code.user_code import UserCode


@serializable()
class CodeHistory(SyftObject):
    # version
    __canonical_name__ = "CodeHistory"
    __version__ = SYFT_OBJECT_VERSION_1

    id: UID
    node_uid: Optional[UID]
    user_verify_key: SyftVerifyKey
    enclave_metadata: Optional[EnclaveMetadata] = None
    user_code_history: Optional[List[UID]] = []
    service_func_name: str
    comment_history: Optional[List[str]] = []
    # comments: Optional[str] = None

    __attr_unique__ = ["service_func_name"]
    __attr_searchable__ = ["user_verify_key"]

    def add_code(self, code: UserCode, comment: Optional[str] = None):
        self.user_code_history.append(code.id)
        if comment is None:
            comment = ""
        self.comment_history.append(comment)


@serializable()
class CodeVersions(SyftObject):
    # version
    __canonical_name__ = "CodeVersions"
    __version__ = SYFT_OBJECT_VERSION_1

    id: UID
    user_code_history: Optional[List[UserCode]] = []
    service_func_name: str
    comment_history: Optional[List[str]] = []

    def _coll_repr_(self):
        return {"Number of versions": len(self.user_code_history)}

    def _repr_html_(self):
        rows = get_repr_values_table(self.user_code_history, True)
        for i, r in enumerate(rows):
            r["Version"] = f"v{i}"
        rows = sorted(rows, key=lambda x: x["Version"], reverse=True)
        return create_table_template(rows, "CodeVersions", table_icon=None)

    def __getitem__(self, key: int):
        return self.user_code_history[key]


@serializable()
class CodeHistoryDict(SyftObject):
    # version
    __canonical_name__ = "CodeHistoryDict"
    __version__ = SYFT_OBJECT_VERSION_1

    id: UID
    code_versions: Optional[Dict[str, CodeVersions]] = {}

    def _repr_html_(self):
        return f"""
            {self.code_versions._repr_html_()}
            """

    def add_func(self, versions: CodeVersions) -> Any:
        self.code_versions[versions.service_func_name] = versions

    def __getitem__(self, name: str) -> Any:
        return self.code_versions[name]

    def __getattr__(self, name: str) -> Any:
        code_versions = object.__getattribute__(self, "code_versions")
        if name in code_versions.keys():
            return code_versions[name]
        return object.__getattribute__(self, name)


@serializable()
class UserHistoryDict(SyftObject):
    # version
    __canonical_name__ = "UserHistoryDict"
    __version__ = SYFT_OBJECT_VERSION_1

    id: UID
    node_uid: UID
    user_dict: Optional[Dict[str, List[str]]] = {}

    __repr_attrs__ = ["available_keys"]

    @property
    def available_keys(self):
        return json.dumps(self.user_dict, sort_keys=True, indent=4)

    def __getitem__(self, key: int):
        api = APIRegistry.api_for(self.node_uid, self.syft_client_verify_key)
        return api.services.code_history.get_history_for_user(key)

    def _repr_html_(self):
        rows = []
        for user, funcs in self.user_dict.items():
            if len(funcs) == 0:
                funcs_str = ""
            else:
                funcs_str = str(["my_func"] * 100)
            rows += [{"user": user, "UserCodes": funcs_str}]
        return create_table_template(rows, "UserCodeHistory", table_icon=None)
