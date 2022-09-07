# stdlib
from typing import Dict
from typing import List
from typing import Optional

# third party
from typing_extensions import final

# relative
from .....common.message import ImmediateSyftMessageWithReply
from .....common.message import ImmediateSyftMessageWithoutReply
from .....common.serde.serializable import serializable
from .....common.uid import UID
from .....io.address import Address


@final
@serializable(recursive_serde=True)
class CreateDatasetMessage(ImmediateSyftMessageWithReply):
    __attr_allowlist__ = [
        "id",
        "address",
        "dataset",
        "metadata",
        "reply_to",
        "platform",
    ]

    def __init__(
        self,
        address: Address,
        dataset: bytes,
        metadata: Dict[str, str],
        reply_to: Address,
        platform: str,
        msg_id: Optional[UID] = None,
    ) -> None:
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.dataset = dataset
        self.metadata = metadata
        self.platform = platform


@final
@serializable(recursive_serde=True)
class GetDatasetMessage(ImmediateSyftMessageWithReply):
    __attr_allowlist__ = ["id", "address", "reply_to", "dataset_id"]

    def __init__(
        self,
        address: Address,
        dataset_id: str,
        reply_to: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.dataset_id = dataset_id


@final
@serializable(recursive_serde=True)
class GetDatasetResponse(ImmediateSyftMessageWithoutReply):
    __attr_allowlist__ = ["id", "address", "metadata"]

    def __init__(
        self,
        address: Address,
        metadata: Dict,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.metadata = metadata


@final
@serializable(recursive_serde=True)
class GetDatasetsMessage(ImmediateSyftMessageWithReply):
    __attr_allowlist__ = ["id", "address", "reply_to"]

    def __init__(
        self,
        address: Address,
        reply_to: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)


@final
@serializable(recursive_serde=True)
class GetDatasetsResponse(ImmediateSyftMessageWithoutReply):
    __attr_allowlist__ = ["id", "address", "metadatas"]

    def __init__(
        self,
        address: Address,
        metadatas: List[Dict],
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.metadatas = metadatas


@final
@serializable(recursive_serde=True)
class UpdateDatasetMessage(ImmediateSyftMessageWithReply):
    __attr_allowlist__ = [
        "id",
        "address",
        "reply_to",
        "reply_to",
        "metadata",
        "dataset_id",
    ]

    def __init__(
        self,
        address: Address,
        dataset_id: str,
        metadata: Dict,
        reply_to: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.metadata = metadata
        self.dataset_id = dataset_id


@final
@serializable(recursive_serde=True)
class DeleteDatasetMessage(ImmediateSyftMessageWithReply):
    __attr_allowlist__ = [
        "id",
        "address",
        "reply_to",
        "reply_to",
        "dataset_id",
        "bin_object_id",
    ]

    def __init__(
        self,
        address: Address,
        dataset_id: str,
        reply_to: Address,
        bin_object_id: Optional[str] = None,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.dataset_id = dataset_id
        self.bin_object_id = bin_object_id
