# third party
from oblv.oblv_client import OblvClient

# relative
from .....core.node.common.node_table.syft_object import SYFT_OBJECT_VERSION_1
from .....core.node.common.node_table.syft_object import SyftObject
from ....common.serde.serializable import serializable
from ....common.uid import UID
from ..credentials import SyftVerifyKey


@serializable(recursive_serde=True)
class EnclaveTransferRequest(SyftObject):
    # version
    __canonical_name__ = "EnclaveTransferRequest"
    __version__ = SYFT_OBJECT_VERSION_1

    # fields
    id: UID = UID()
    deployment_id: str
    data_id: UID
    user_verify_key: SyftVerifyKey
    status: bool = False
    created_at: str
    reviewed_at: str = ""
    reviewed_by: str = ""
    reason: str = ""
    oblv_client: OblvClient

    # serde / storage rules
    __attr_state__ = [
        "id",
        "deployment_id",
        "data_id",
        "user_verify_key",
        "status",
        "created_at",
        "reviewed_at",
        "reason",
        "reviewed_by",
        "oblv_client",
    ]
    __attr_searchable__ = [
        "id",
        "deployment_id",
        "data_id",
        "user_verify_key",
        "status",
    ]
    __attr_unique__ = ["id"]
