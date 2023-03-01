# stdlib
from typing import Any
from typing import Optional

# third party
from result import Err
from result import Ok
from result import Result

# relative
from ....common.serde.serializable import serializable
from ....common.uid import UID
from ..document_store import BaseStash
from ..document_store import DocumentStore
from ..document_store import PartitionSettings
from ..document_store import QueryKeys
from ..document_store import UIDPartitionKey
from .enclave_transfer_request import EnclaveTransferRequest


@serializable(recursive_serde=True)
class EnclaveTransferRequestStash(BaseStash):
    object_type = EnclaveTransferRequest
    settings: PartitionSettings = PartitionSettings(
        name=EnclaveTransferRequest.__canonical_name__,
        object_type=EnclaveTransferRequest,
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    def check_type(self, obj: Any, type_: type) -> Result[Any, str]:
        return (
            Ok(obj)
            if isinstance(obj, type_)
            else Err(f"{type(obj)} does not match required type: {type_}")
        )

    def set(
        self, enclave_transfer_request: EnclaveTransferRequest
    ) -> Result[EnclaveTransferRequest, Err]:
        return self.check_type(enclave_transfer_request, self.object_type).and_then(
            super().set
        )

    def get_by_uid(self, uid: UID) -> Result[Optional[EnclaveTransferRequest], str]:
        qks = QueryKeys(qks=[UIDPartitionKey.with_obj(uid)])
        return Ok(self.query_one(qks=qks))

    def update(
        self, request: EnclaveTransferRequest
    ) -> Result[EnclaveTransferRequest, str]:
        return self.check_type(request, self.object_type).and_then(super().update)
