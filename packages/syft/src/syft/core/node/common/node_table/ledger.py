# third party
from sqlalchemy import Column
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import String

# relative
from . import Base


class Ledger(Base):
    __tablename__ = "ledger"

    id = Column(Integer(), primary_key=True, autoincrement=True)
    entity_name = Column(String(256), ForeignKey("entity.name", ondelete="CASCADE"))
    mechanism_id = Column(Integer(), ForeignKey("mechanism.id", ondelete="CASCADE"))
