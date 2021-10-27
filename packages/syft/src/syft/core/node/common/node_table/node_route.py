# third party
from sqlalchemy import Boolean
from sqlalchemy import Column
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import String

# relative
from . import Base


class NodeRoute(Base):
    __tablename__ = "node_route"

    id = Column(Integer(), primary_key=True, autoincrement=True)
    node_id = Column(Integer, ForeignKey("node.id"))
    host_or_ip = Column(String(255))
    is_vpn = Column(Boolean(), default=False)
