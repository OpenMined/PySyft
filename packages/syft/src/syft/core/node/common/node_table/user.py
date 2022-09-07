# future
from __future__ import annotations

# stdlib
from typing import Optional

# third party
from sqlalchemy import Column
from sqlalchemy import Float
from sqlalchemy import Integer
from sqlalchemy import String

# relative
from . import Base
from .syft_object import SyftObject


class UserApplication(Base):
    __tablename__ = "syft_application"

    id = Column(Integer(), primary_key=True, autoincrement=True)
    email = Column(String(255))
    name = Column(String(255), default="")
    hashed_password = Column(String(512))
    salt = Column(String(255))
    # daa_pdf = Column(Integer, ForeignKey("daa_pdf.id"))
    status = Column(String(255), default="pending")
    added_by = Column(String(2048))
    website = Column(String(2048))
    institution = Column(String(2048))
    budget = Column(Float(), default=0.0)

    def __str__(self) -> str:
        return (
            f"<User Application id: {self.id}, email: {self.email}, name: {self.name}"
            f"status: {self.status}>"
        )


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
    daa_pdf: Optional[str]
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
