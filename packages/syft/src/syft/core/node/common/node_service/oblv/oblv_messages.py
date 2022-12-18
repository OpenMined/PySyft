# stdlib
from typing import Optional
from typing import Union

# third party
from oblv import OblvClient
from typing_extensions import final

# relative
from .....common.message import ImmediateSyftMessageWithReply
from .....common.message import ImmediateSyftMessageWithoutReply
from .....common.serde.serializable import serializable
from .....common.uid import UID
from .....io.address import Address
from .....tensor.autodp.phi_tensor import TensorWrappedPhiTensorPointer


@serializable(recursive_serde=True)
@final
class SyftOblvClient:
    __attr_allowlist__ = (
        "token",
        "oblivious_user_id",
        "cookies",
        "headers",
        "timeout",
        "verify_ssl",
    )

    @classmethod
    def from_client(cls, input: OblvClient):
        obj = SyftOblvClient()
        obj.token = input.token
        obj.oblivious_user_id = input.oblivious_user_id
        obj.cookies = input.cookies
        obj.headers = input.headers
        obj.timeout = input.timeout
        obj.verify_ssl = input.verify_ssl
        return obj

    def __init__(
        self,
        token: Optional[str] = None,
        oblivious_user_id: Optional[str] = None,
        cookies: Optional[dict] = {},
        headers: Optional[dict] = {},
        timeout: float = 20,
        verify_ssl: bool = True,
    ):
        super().__init__()
        self.token = token
        self.oblivious_user_id = oblivious_user_id
        self.cookies = cookies
        self.headers = headers
        self.timeout = timeout
        self.verify_ssl = verify_ssl


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
class PublishDatasetMessage(ImmediateSyftMessageWithReply):
    __attr_allowlist__ = [
        "id",
        "address",
        "reply_to",
        "dataset_id",
        "deployment_id",
        "client",
    ]

    def __init__(
        self,
        address: Address,
        reply_to: Address,
        deployment_id: str,
        client: SyftOblvClient,
        dataset_id: Union[str, TensorWrappedPhiTensorPointer] = "",
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.deployment_id = deployment_id
        self.client = client
        if type(dataset_id) == TensorWrappedPhiTensorPointer:
            self.dataset_id = dataset_id.id_at_location.to_string()
        else:
            self.dataset_id = dataset_id


@serializable(recursive_serde=True)
@final
class CheckEnclaveConnectionMessage(ImmediateSyftMessageWithReply):
    __attr_allowlist__ = ["id", "address", "reply_to", "deployment_id", "client"]

    def __init__(
        self,
        address: Address,
        reply_to: Address,
        deployment_id: str,
        client: SyftOblvClient,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.deployment_id = deployment_id
        self.client = client


@serializable(recursive_serde=True)
@final
class PublishDatasetResponse(ImmediateSyftMessageWithoutReply):
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
class PublishApprovalMessage(ImmediateSyftMessageWithoutReply):
    __attr_allowlist__ = ["id", "address", "result_id", "deployment_id", "client"]

    def __init__(
        self,
        address: Address,
        deployment_id: str,
        result_id: str,
        client: SyftOblvClient,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.result_id = result_id
        self.client = client
        self.deployment_id = deployment_id


@serializable(recursive_serde=True)
@final
class DeductBudgetMessage(ImmediateSyftMessageWithoutReply):
    __attr_allowlist__ = [
        "id",
        "address",
        "result_id",
        "deployment_id",
        "client",
        "budget_to_deduct",
    ]

    def __init__(
        self,
        address: Address,
        deployment_id: str,
        result_id: str,
        budget_to_deduct: float,
        client: SyftOblvClient,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.result_id = result_id
        self.budget_to_deduct = budget_to_deduct
        self.client = client
        self.deployment_id = deployment_id
