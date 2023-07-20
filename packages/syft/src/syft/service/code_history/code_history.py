# stdlib
from typing import List
from typing import Optional

# relative
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
    user_code_history: Optional[List[UID]] = []
    service_func_name: str

    __attr_unique__ = ["service_func_name"]

    def add_code(self, code: UserCode):
        self.user_code_history.append(code.id)


# TODO: Fix Multiple users can passing the same name for their code.
