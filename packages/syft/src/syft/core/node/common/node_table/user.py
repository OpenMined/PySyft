# future
from __future__ import annotations

# stdlib
from typing import Optional

# relative
from .syft_object import SyftObject


class NoSQLSyftUser(SyftObject):
    # version
    __canonical_name__ = "SyftUser"
    __version__ = 1

    # fields
    email: str
    name: str
    budget: float
    hashed_password: str
    salt: str
    private_key: str
    verify_key: str
    role: dict
    added_by: Optional[str]
    website: Optional[str]
    institution: Optional[str]
    daa_pdf: Optional[bytes]
    created_at: Optional[str]
    id_int: Optional[int]

    # serde / storage rules
    __attr_state__ = [
        "email",
        "name",
        "budget",
        "hashed_password",
        "salt",
        "private_key",
        "verify_key",
        "role",
        "added_by",
        "website",
        "institution",
        "daa_pdf",
        "created_at",
        "id_int",
    ]
    __attr_searchable__ = ["email", "verify_key", "id_int"]
    __attr_unique__ = ["email"]


class NoSQLUserApplication(SyftObject):
    # version
    __canonical_name__ = "UserApplication"
    __version__ = 1

    # fields
    id_int: int
    email: str
    name: str
    hashed_password: str
    salt: str
    daa_pdf: Optional[bytes]
    status: str = "pending"
    added_by: Optional[str]
    website: Optional[str]
    institution: Optional[str]
    budget: float = 0.0

    # serde / storage rules
    __attr_state__ = [
        "id_int",
        "email",
        "name",
        "hashed_password",
        "salt",
        "daa_pdf",
        "status",
        "added_by",
        "website",
        "institution",
        "budget",
    ]
    __attr_searchable__ = ["email", "id_int"]
    __attr_unique__ = ["email"]
