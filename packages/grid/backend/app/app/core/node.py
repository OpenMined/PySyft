# syft absolute
from syft import Domain
from syft.core.node.common.tables import Base
from syft.core.node.common.tables.utils import seed_db

# grid absolute
from app.db.session import SessionLocal
from app.db.session import engine

domain = Domain("my domain", db_engine=engine)
Base.metadata.create_all(engine)
seed_db(SessionLocal())
