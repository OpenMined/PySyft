# third party
from sqlalchemy import Column
from sqlalchemy import DateTime
from sqlalchemy import Float
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import String

# relative
from . import Base


class UserApplication(Base):
    __tablename__ = "syft_application"

    id = Column(Integer(), primary_key=True, autoincrement=True)
    email = Column(String(255))
    name = Column(String(255), default="")
    hashed_password = Column(String(512))
    salt = Column(String(255))
    daa_pdf = Column(Integer, ForeignKey("daa_pdf.id"))
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


class SyftUser(Base):
    __tablename__ = "syft_user"

    id = Column(Integer(), primary_key=True, autoincrement=True)
    email = Column(String(255))
    name = Column(String(255), default="")
    budget = Column(Float(), default=0.0)
    hashed_password = Column(String(512))
    salt = Column(String(255))
    private_key = Column(String(2048))
    verify_key = Column(String(2048))
    role = Column(Integer, ForeignKey("role.id"))
    added_by = Column(String(2048))
    website = Column(String(2048))
    institution = Column(String(2048))
    daa_pdf = Column(Integer, ForeignKey("daa_pdf.id"))
    created_at = Column(DateTime())

    def __str__(self) -> str:
        return (
            f"<User id: {self.id}, email: {self.email}, name: {self.name}"
            f"role: {self.role}>"
        )


def create_user(
    email: str,
    hashed_password: str,
    salt: str,
    private_key: str,
    role: int,
    name: str = "",
    budget: float = 0.0,
) -> SyftUser:
    new_user = SyftUser(
        email=email,
        hashed_password=hashed_password,
        salt=salt,
        private_key=private_key,
        role=role,
        name=name,
        budget=budget,
    )
    return new_user
