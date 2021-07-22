# third party
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import String

# relative
from . import Base


class SetupConfig(Base):
    __tablename__ = "setup"

    id = Column(Integer(), primary_key=True, autoincrement=True)
    domain_name = Column(String(255), default="")
    node_id = Column(String(32), default="")

    def __str__(self):
        return f"<Domain Name: {self.domain_name}>"


def create_setup(**kwargs):
    new_setup = SetupConfig(**kwargs)
    return new_setup
