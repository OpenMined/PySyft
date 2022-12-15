# stdlib
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

# relative
from ....common.uid import UID
from .syft_object import SyftObject


class NoSQLRequest(SyftObject):
    # version
    __canonical_name__ = "Request"
    __version__ = 1

    # fields
    id: UID
    date: str
    user_id: int
    user_name: str
    user_email: str
    user_role: str
    user_budget: Optional[float]
    institution: Optional[str]
    website: Optional[str]
    object_id: UID
    reason: str
    status: str = "pending"
    request_type: str
    verify_key: str
    object_type: str
    tags: Optional[List[str]] = []
    updated_on: Optional[str]
    reviewer_name: Optional[str]
    reviewer_role: Optional[str]
    reviewer_comment: Optional[str]
    requested_budget: Optional[float]
    current_budget: Optional[float]

    # serde / storage rules
    __attr_state__ = [
        "id",
        "date",
        "user_id",
        "user_name",
        "user_email" "user_role",
        "user_budget",
        "institution",
        "website",
        "object_id",
        "reason",
        "status",
        "request_type",
        "verify_key",
        "object_type",
        "tags",
        "updated_on",
        "reviewer_name",
        "reviewer_role",
        "reviewer_comment",
        "requested_budget",
        "current_budget",
    ]
    __attr_searchable__: List[str] = ["status", "verify_key", "request_type"]
    __attr_unique__: List[str] = []

    def to_dict(self) -> Dict[Any, Any]:
        attr_dict = super().to_dict()
        attr_dict["id"] = attr_dict["id"].to_string()
        attr_dict["object_id"] = attr_dict["object_id"].to_string()
        return attr_dict
