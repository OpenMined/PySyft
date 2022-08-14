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


@serializable(recursive_serde=True)
@final
class CreateUserMessage(ImmediateSyftMessageWithReply):
    __attr_allowlist__ = [
        "id",
        "address",
        "email",
        "password",
        "role",
        "name",
        "institution",
        "website",
        "daa_pdf",
        "reply_to",
        "budget",
    ]

    def __init__(
        self,
        address: Address,
        name: str,
        email: str,
        password: str,
        reply_to: Address,
        role: Optional[str] = "",
        website: str = "",
        institution: str = "",
        daa_pdf: Optional[bytes] = b"",
        msg_id: Optional[UID] = None,
        budget: Optional[float] = 0.0,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.email = email
        self.password = password
        self.role = role
        self.name = name
        self.daa_pdf = daa_pdf
        self.website = website
        self.institution = institution
        self.budget = budget


@serializable(recursive_serde=True)
@final
class GetUserMessage(ImmediateSyftMessageWithReply):
    __attr_allowlist__ = ["id", "address", "reply_to", "user_id"]

    def __init__(
        self,
        address: Address,
        user_id: int,
        reply_to: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.user_id = user_id


@serializable(recursive_serde=True)
@final
class ProcessUserCandidateMessage(ImmediateSyftMessageWithReply):
    __attr_allowlist__ = ["id", "address", "reply_to", "candidate_id", "status"]

    def __init__(
        self,
        address: Address,
        candidate_id: int,
        status: str,
        reply_to: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.candidate_id = candidate_id
        self.status = status


@serializable(recursive_serde=True)
@final
class GetUserResponse(ImmediateSyftMessageWithoutReply):
    __attr_allowlist__ = ["id", "address", "content"]

    def __init__(
        self,
        address: Address,
        content: Dict,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.content = content


@serializable(recursive_serde=True)
@final
class GetUsersMessage(ImmediateSyftMessageWithReply):
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
class GetCandidatesMessage(ImmediateSyftMessageWithReply):
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
class GetCandidatesResponse(ImmediateSyftMessageWithoutReply):
    __attr_allowlist__ = ["id", "address", "content"]

    def __init__(
        self,
        address: Address,
        content: List[Dict],
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.content = content


@serializable(recursive_serde=True)
@final
class GetUsersResponse(ImmediateSyftMessageWithoutReply):
    __attr_allowlist__ = ["id", "address", "content"]

    def __init__(
        self,
        address: Address,
        content: List[Dict],
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.content = content


@serializable(recursive_serde=True)
@final
class UpdateUserMessage(ImmediateSyftMessageWithReply):
    __attr_allowlist__ = [
        "id",
        "address",
        "user_id",
        "email",
        "password",
        "new_password",
        "role",
        "groups",
        "budget",
        "institution",
        "website",
        "name",
        "reply_to",
    ]

    def __init__(  # nosec
        self,
        address: Address,
        reply_to: Address,
        user_id: Optional[int] = 0,
        msg_id: Optional[UID] = None,
        email: Optional[str] = "",
        password: Optional[str] = "",
        new_password: Optional[str] = "",
        role: Optional[str] = "",
        groups: Optional[str] = "",
        budget: Optional[float] = None,
        name: Optional[str] = "",
        institution: Optional[str] = "",
        website: Optional[str] = "",
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.user_id = user_id
        self.email = email
        self.password = password
        self.role = role
        self.groups = groups
        self.name = name
        self.budget = budget
        self.institution = institution
        self.website = website
        self.new_password = new_password


@serializable(recursive_serde=True)
@final
class DeleteUserMessage(ImmediateSyftMessageWithReply):
    __attr_allowlist__ = ["id", "address", "reply_to", "user_id"]

    def __init__(
        self,
        address: Address,
        user_id: int,
        reply_to: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.user_id = user_id


@serializable(recursive_serde=True)
@final
class SearchUsersMessage(ImmediateSyftMessageWithReply):
    __attr_allowlist__ = [
        "id",
        "address",
        "reply_to",
        "email",
        "role",
        "groups",
        "name",
    ]

    def __init__(
        self,
        address: Address,
        reply_to: Address,
        msg_id: Optional[UID] = None,
        email: Optional[str] = "",
        role: Optional[str] = "",
        groups: Optional[str] = "",
        name: Optional[str] = "",
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.email = email
        self.role = role
        self.groups = groups
        self.name = name


@serializable(recursive_serde=True)
@final
class SearchUsersResponse(ImmediateSyftMessageWithoutReply):
    __attr_allowlist__ = ["address", "content", "id"]

    def __init__(
        self,
        address: Address,
        content: List[Dict],
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.content = content
