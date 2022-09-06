# third party
from sqlalchemy import Boolean
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import String

# relative
from . import Base


# this table holds the list of known nodes usually peer domains
class Node(Base):
    __tablename__ = "node"

    id = Column(Integer(), primary_key=True, autoincrement=True)
    node_uid = Column(String(255))
    node_name = Column(String(255))
    node_type = Column(String(255))
    verify_key = Column(String(2048))
    keep_connected = Column(Boolean(), default=True)
