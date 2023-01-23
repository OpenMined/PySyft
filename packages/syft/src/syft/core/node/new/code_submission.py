# stdlib
from datetime import datetime
from enum import Enum
from typing import Optional

# third party
from action_object import ActionObjectPointer
from nacl.signing import VerifyKey

# relative
from ..common import UID


class CodeSubmissionEnum(str, Enum):
    ACCEPT = "ACCEPTED"
    DENY = "REJECTED"
    PENDING = "PENDING"


class CodeSubmissionObject:
    def __init__(
        self,
        verify_key: VerifyKey,
        result_uid: UID,
        pointer_type: ActionObjectPointer,
        code: str,
        reason: str,
    ):
        # Provided by Data Scientist
        self.verify_key = verify_key
        self.result_uid = result_uid
        self.pointer_type = pointer_type
        self.code = code
        self.reason = reason

        # Inferred
        self.status = CodeSubmissionEnum.PENDING
        self.created_at = datetime.now()

        # Provided by Data Owner
        self.response: Optional[str] = "Awaiting response"
        self.reviewed_at: Optional[datetime] = None

        # Probably not wise to put this on the same object?
        # self.reviewer_verify_key: Optional[VerifyKey] = None

    def __str__(self):
        # This could be gigantic
        return ""

    def __repr__(self):
        return ""


class CodeSubmissionListObject:
    def __init__(self):
        pass
