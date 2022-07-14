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
    host_or_ip = Column(String(255), default="")
    is_vpn = Column(Boolean(), default=False)
    private = Column(Boolean(), default=False)
    protocol = Column(String(255), default="http")
    port = Column(Integer(), default=80)
    vpn_endpoint = Column(String(255), default="")
    vpn_key = Column(String(255), default="")
