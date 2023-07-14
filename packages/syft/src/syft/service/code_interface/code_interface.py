# stdlib
from typing import Dict
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
class CodeInterface(SyftObject):
    # version
    __canonical_name__ = "CodeInterface"
    __version__ = SYFT_OBJECT_VERSION_1

    id: UID
    node_uid: Optional[UID]
    user_verify_key: SyftVerifyKey
    enclave_metadata: Optional[EnclaveMetadata] = None
    user_code_mapping: Optional[Dict[str, UID]] = {}
    service_func_name: str

    __attr_unique__ = ["service_func_name"]

    def add_code(self, code: UserCode):
        if code.service_func_name in self.user_code_mapping:
            new_version = (
                max(self.user_code_mapping[code.service_func_name].keys(), default=0)
                + 1
            )
        else:
            self.user_code_mapping[code.service_func_name] = {}
            new_version = 1

        self.user_code_mapping[code.service_func_name][new_version] = code.id
