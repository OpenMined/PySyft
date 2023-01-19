# stdlib
from typing import Dict
from typing import List as TypeList
from typing import Optional

# third party
from nacl.signing import SigningKey

# relative
from .....common.message import ImmediateSyftMessageWithReply
from .....common.message import ImmediateSyftMessageWithoutReply
from .....common.serde.serializable import serializable
from .....common.uid import UID


@serializable(recursive_serde=True)
class GetSetUpMessage(ImmediateSyftMessageWithReply):
    __attr_allowlist__ = ["address", "id", "reply_to"]

    def __init__(
        self,
        address: UID,
        reply_to: UID,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)


@serializable(recursive_serde=True)
class GetSetUpResponse(ImmediateSyftMessageWithoutReply):
    __attr_allowlist__ = ["address", "id", "content"]

    def __init__(
        self,
        address: UID,
        content: Dict,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.content = content


@serializable(recursive_serde=True)
class CreateInitialSetUpMessage(ImmediateSyftMessageWithReply):
    __attr_allowlist__ = [
        "address",
        "id",
        "name",
        "email",
        "password",
        "domain_name",
        "budget",
        "reply_to",
        "signing_key",
    ]

    def __init__(
        self,
        address: UID,
        name: str,
        email: str,
        password: str,
        domain_name: str,
        budget: float,
        reply_to: UID,
        signing_key: SigningKey,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.name = name
        self.email = email
        self.password = password
        self.domain_name = domain_name
        self.budget = budget
        self.signing_key = signing_key


@serializable(recursive_serde=True)
class UpdateSetupMessage(ImmediateSyftMessageWithReply):
    __attr_allowlist__ = [
        "id",
        "address",
        "domain_name",
        "contact",
        "daa",
        "description",
        "daa_document",
        "tags",
        "reply_to",
    ]

    def __init__(
        self,
        address: UID,
        domain_name: str,
        description: str,
        daa: bool,
        contact: str,
        reply_to: UID,
        daa_document: Optional[bytes] = b"",
        tags: Optional[TypeList] = None,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.daa = daa
        self.contact = contact
        self.description = description
        self.domain_name = domain_name
        self.daa_document = daa_document
        self.tags = tags if tags is not None else []


@serializable(recursive_serde=True)
class UpdateSetupResponse(ImmediateSyftMessageWithoutReply):
    __attr_allowlist__ = ["id", "address", "content"]

    def __init__(
        self,
        address: UID,
        content: Dict,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.content = content
