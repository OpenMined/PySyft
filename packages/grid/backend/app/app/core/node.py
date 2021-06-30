
from syft.core.node.common.tables import Base
from syft.core.node.common.tables.utils import seed_db
from syft import Domain
from app.db.session import engine
from app.db.session import SessionLocal

domain = Domain("my domain", db_engine=engine)
Base.metadata.create_all(engine)
seed_db(SessionLocal())