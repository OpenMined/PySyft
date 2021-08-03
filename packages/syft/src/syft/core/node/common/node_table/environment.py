# stdlib
from datetime import datetime

# third party
from sqlalchemy import Column
from sqlalchemy import DateTime
from sqlalchemy import Integer
from sqlalchemy import String

# relative
from . import Base

states = {"creating": 0, "failed": 1, "success": 2, "destroyed": 3}


class Environment(Base):
    __tablename__ = "environment"

    id = Column(Integer(), primary_key=True, autoincrement=True)
    state = Column(Integer)
    provider = Column(String(255))
    region = Column(String(255))
    instance_type = Column(String(255))
    address = Column(String(255), default="0.0.0.0")  # nosec
    syft_address = Column(String(255), default="")  # TODO
    created_at = Column(DateTime, default=datetime.now())
    destroyed_at = Column(DateTime, default=datetime.now())

    def __str__(self) -> str:
        return (
            f"<Group id: {self.id}, state: {self.state}, address: {self.address}, syft_address: "
            f"{self.syft_address}, provider: {self.provider}, region: {self.region}, instance_type: "
            f"{self.instance_type}, created_at: {self.created_at}, destroyed_at: {self.destroyed_at}>"
        )
