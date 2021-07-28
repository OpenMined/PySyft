# third party
from sqlalchemy import Column
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import String

# relative
from . import Base


class SyftUser(Base):
    __tablename__ = "syft_user"

    id = Column(Integer(), primary_key=True, autoincrement=True)
    email = Column(String(255))
    name = Column(String(255), default="")
    hashed_password = Column(String(512))
    salt = Column(String(255))
    private_key = Column(String(2048))
    verify_key = Column(String(2048))
    role = Column(Integer, ForeignKey("role.id"))

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
) -> SyftUser:
    new_user = SyftUser(
        email=email,
        hashed_password=hashed_password,
        salt=salt,
        private_key=private_key,
        role=role,
        name=name,
    )
    return new_user
