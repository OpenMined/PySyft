# stdlib
from typing import List, Dict, Tuple
from typing import Optional

# relative
from ...client.api import SyftAPI
from ...client.enclave_client import EnclaveMetadata
from ...serde.serializable import serializable
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ...types.syft_object import SyftVerifyKey
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
    user_code_history: Optional[List[Tuple[UID, str]]] = []
    service_func_name: str
    comment_history: Optional[Dict[UID, str]] = {}
    # comments: Optional[str] = None

    __attr_unique__ = ["service_func_name"]

    def add_code(self, code: UserCode, comment: str):
        self.user_code_history.append((code.id, comment))

    # def __getitem__(self, key: int):  
    #     api = APIRegistry.api_for(
    #         self.node_uid,
    #         self.user_verify_key,
    #     )
    #     return api.services.code.get_by_id(self.user_code_history[key])


@serializable()
class CodeVersions:
    user_code_history: Optional[List[UserCode]] = []
    service_func_name: str
    api: SyftAPI

    def __init__(self, user_code_history=None, service_func_name="") -> None:
        self.user_code_history = user_code_history
        self.service_func_name = service_func_name

    def __getitem__(self, key: int):
        return self.user_code_history[key]