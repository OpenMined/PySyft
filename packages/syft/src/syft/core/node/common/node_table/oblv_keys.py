# third party
from sqlalchemy import Column
from sqlalchemy import LargeBinary
from sqlalchemy import String

# relative
from . import Base


class OblvKeys(Base):
    __tablename__ = "oblv_keys"

    id = Column(String(256), primary_key=True)
    public_key = Column(LargeBinary(3072))
    private_key = Column(LargeBinary(3072))
