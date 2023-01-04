# stdlib
from typing import Dict

# relative
from .syft_object import SyftObject


class NoSQLTask(SyftObject):
    # version
    __canonical_name__ = "Task"
    __version__ = 1

    # fields
    uid: str
    user: str
    load_vars: Dict[str, str]
    owner: Dict[str, str]
    code: str
    status: str
    created_at: str
    updated_at: str
    reviewed_by: str
    execution: Dict[str, str]
    saved_vars: Dict[str, str]
    reason: str = ""

    # serde / storage rules
    __attr_state__ = [
        "uid",
        "code",
        "user",
        "status",
        "load_vars",
        "saved_vars",
        "created_at",
        "updated_at",
        "reviewed_by",
        "execution",
        "reason",
    ]

    __attr_searchable__ = ["uid", "status", "user"]
    __attr_unique__ = ["uid"]
