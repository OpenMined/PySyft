from ...types.syft_object import SyftObject, SYFT_OBJECT_VERSION_1, SyftVerifyKey
from ...types.uid import UID
from typing import Optional, Dict
from ..metadata.node_metadata import EnclaveMetadata
from ..code.user_code import UserCode

class CodeInterface(SyftObject):
    # version
    __canonical_name__ = "CodeInterface"
    __version__ = SYFT_OBJECT_VERSION_1

    id: UID
    node_uid: Optional[UID]
    user_verify_key: SyftVerifyKey
    enclave_metadata: Optional[EnclaveMetadata] = None
    user_code_mapping: Optional[Dict[str, UserCode]] = None
    service_func_name: str









