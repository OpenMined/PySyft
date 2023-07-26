# stdlib
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
from ...types.syft_object import list_dict_repr_html
from ...types.uid import UID
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

    # def __getitem__(self, key: int):
    #     api = APIRegistry.api_for(
    #         self.node_uid,
    #         self.user_verify_key,
    #     )
    #     return api.services.code.get_by_id(self.user_code_history[key])


@serializable()
class CodeVersions(SyftObject):
    # version
    __canonical_name__ = "CodeVersions"
    __version__ = SYFT_OBJECT_VERSION_1

    id: UID
    user_code_history: Optional[List[UserCode]] = []
    service_func_name: str
    comment_history: Optional[List[str]] = []

    def _repr_html_(self):
        return self.user_code_history._repr_html_()

    # def __init__(self, user_code_history=None, service_func_name="", comment_history = []) -> None:
    #     self.user_code_history = user_code_history
    #     self.service_func_name = service_func_name
    #     self.comment_history = comment_history

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
        return self.code_versions._repr_html_()

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

    def _repr_html_(self):
        return list_dict_repr_html(self.user_dict)

    def __getitem__(self, key: int):
        api = APIRegistry.api_for(self.node_uid, self.syft_client_verify_key)
        return api.services.code_history.get_history_for_user(key)
