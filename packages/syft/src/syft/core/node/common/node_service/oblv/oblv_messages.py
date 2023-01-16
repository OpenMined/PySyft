# stdlib
from typing import Optional
from typing import Union

# third party
from oblv import OblvClient
from typing_extensions import final

# relative
from ......core.pointer.pointer import Pointer
from .....common.message import ImmediateSyftMessageWithReply
from .....common.message import ImmediateSyftMessageWithoutReply
from .....common.serde.serializable import serializable
from .....common.uid import UID
from .....io.address import Address
from .....tensor.smpc.utils import ispointer


@serializable(recursive_serde=True)
@final
class CreateKeyPairMessage(ImmediateSyftMessageWithReply):
    __attr_allowlist__ = ["id", "address", "reply_to"]

    def __init__(
        self,
        address: Address,
        reply_to: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)


@serializable(recursive_serde=True)
@final
class CreateKeyPairResponse(ImmediateSyftMessageWithoutReply):
    __attr_allowlist__ = ["id", "address", "response"]

    def __init__(
        self,
        address: Address,
        response: str,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.response = response


@serializable(recursive_serde=True)
@final
class GetPublicKeyMessage(ImmediateSyftMessageWithReply):
    __attr_allowlist__ = ["id", "address", "reply_to"]

    def __init__(
        self,
        address: Address,
        reply_to: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)


@serializable(recursive_serde=True)
@final
class GetPublicKeyResponse(ImmediateSyftMessageWithoutReply):
    __attr_allowlist__ = ["id", "address", "response"]

    def __init__(
        self,
        address: Address,
        response: str = "",
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.response = response


@serializable(recursive_serde=True)
@final
class TransferDatasetMessage(ImmediateSyftMessageWithReply):
    __attr_allowlist__ = [
        "id",
        "address",
        "reply_to",
        "dataset_id",
        "deployment_id",
        "oblv_client",
    ]

    def __init__(
        self,
        address: Address,
        reply_to: Address,
        deployment_id: str,
        oblv_client: OblvClient,
        dataset_id: Union[str, Pointer] = "",
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.deployment_id = deployment_id
        self.oblv_client = oblv_client
        if ispointer(dataset_id):
            self.dataset_id = dataset_id.id_at_location.to_string()
        else:
            self.dataset_id = dataset_id


@serializable(recursive_serde=True)
@final
class TransferDatasetResponse(ImmediateSyftMessageWithoutReply):
    __attr_allowlist__ = ["id", "address", "dataset_id"]

    def __init__(
        self,
        address: Address,
        dataset_id: str = "",
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.dataset_id = dataset_id


@serializable(recursive_serde=True)
@final
class CheckEnclaveConnectionMessage(ImmediateSyftMessageWithReply):
    __attr_allowlist__ = ["id", "address", "reply_to", "deployment_id", "oblv_client"]

    def __init__(
        self,
        address: Address,
        reply_to: Address,
        deployment_id: str,
        oblv_client: OblvClient,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.deployment_id = deployment_id
        self.oblv_client = oblv_client


@serializable(recursive_serde=True)
@final
class PublishApprovalMessage(ImmediateSyftMessageWithoutReply):
    __attr_allowlist__ = ["id", "address", "result_id", "deployment_id", "oblv_client"]

    def __init__(
        self,
        address: Address,
        deployment_id: str,
        result_id: str,
        oblv_client: OblvClient,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.result_id = result_id
        self.oblv_client = oblv_client
        self.deployment_id = deployment_id


@serializable(recursive_serde=True)
@final
class DeductBudgetMessage(ImmediateSyftMessageWithoutReply):
    __attr_allowlist__ = [
        "id",
        "address",
        "result_id",
        "deployment_id",
        "oblv_client",
        "budget_to_deduct",
    ]

    def __init__(
        self,
        address: Address,
        deployment_id: str,
        result_id: str,
        budget_to_deduct: float,
        oblv_client: OblvClient,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.result_id = result_id
        self.budget_to_deduct = budget_to_deduct
        self.oblv_client = oblv_client
        self.deployment_id = deployment_id
