# stdlib
from typing import Dict

# relative
from ....common.serde.serializable import serializable
from .syft_object import SyftObject


@serializable(recursive_serde=True)
class NoSQLPolicy(SyftObject):
    # version
    __canonical_name__ = "Policy"
    __version__ = 1

    # fields
    uid: str
    user: str
    code: str
    name: str
    init_policy_args: Dict[str, str]
    apply_policy_args: Dict[str, str]
    status: str
    created_at: str
    reviewed_by: str
    reason: str = ""

    # serde / storage rules
    __attr_state__ = [
    "uid",
    "user",
    "code",
    "name",
    "init_policy_args",
    "apply_policy_args",
    "state",
    "status",
    "created_at",
    "reviewed_by",
    "reason",
    ]

    __attr_searchable__ = ["uid", "status", "user", "name"]
    __attr_unique__ = ["uid"]
