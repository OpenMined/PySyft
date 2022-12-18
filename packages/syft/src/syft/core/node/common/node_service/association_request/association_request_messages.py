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
class SendAssociationRequestMessage(ImmediateSyftMessageWithReply):
    __attr_allowlist__ = ["id", "address", "source", "target", "reply_to", "metadata"]

    def __init__(
        self,
        source: str,
        target: str,
        address: Address,
        reply_to: Address,
        metadata: Dict[str, str],
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.source = source
        self.target = target
        self.metadata = metadata


@final
@serializable(recursive_serde=True)
class ReceiveAssociationRequestMessage(ImmediateSyftMessageWithoutReply):
    __attr_allowlist__ = ["id", "address", "response", "metadata", "source", "target"]

    def __init__(
        self,
        address: Address,
        source: str,
        target: str,
        metadata: Dict[str, str],
        msg_id: Optional[UID] = None,
        response: str = "",
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.metadata = metadata
        self.response = response
        self.source = source
        self.target = target


@final
@serializable(recursive_serde=True)
class RespondAssociationRequestMessage(ImmediateSyftMessageWithReply):
    __attr_allowlist__ = ["id", "address", "response", "reply_to", "source", "target"]

    def __init__(
        self,
        address: Address,
        response: str,
        reply_to: Address,
        source: str,
        target: str,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.response = response
        self.source = source
        self.target = target


@final
@serializable(recursive_serde=True)
class GetAssociationRequestMessage(ImmediateSyftMessageWithReply):
    __attr_allowlist__ = ["id", "address", "association_id", "reply_to"]

    def __init__(
        self,
        address: Address,
        association_id: int,
        reply_to: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.association_id = association_id


@final
@serializable(recursive_serde=True)
class GetAssociationRequestResponse(ImmediateSyftMessageWithoutReply):
    __attr_allowlist__ = ["id", "address", "content", "source", "target"]

    def __init__(
        self,
        address: Address,
        content: Dict,
        source: str,
        target: str,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.content = content
        self.source = source
        self.target = target


@final
@serializable(recursive_serde=True)
class GetAssociationRequestsMessage(ImmediateSyftMessageWithReply):
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
class GetAssociationRequestsResponse(ImmediateSyftMessageWithoutReply):
    __attr_allowlist__ = ["id", "address", "content"]

    def __init__(
        self,
        address: Address,
        content: List[Dict],
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.content = content


@final
@serializable(recursive_serde=True)
class DeleteAssociationRequestMessage(ImmediateSyftMessageWithReply):
    __attr_allowlist__ = ["id", "address", "reply_to", "association_id"]

    def __init__(
        self,
        address: Address,
        association_id: int,
        reply_to: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.association_id = association_id
