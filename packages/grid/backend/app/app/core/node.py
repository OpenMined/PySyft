# syft absolute
from syft import Domain
from syft import Network
from syft.core.node.common.tables import Base
from syft.core.node.common.tables.utils import seed_db

# grid absolute
from app.db.session import SessionLocal
from app.db.session import engine

domain = Network("Domain", db_engine=engine)
Base.metadata.create_all(engine)

if len(domain.setup):  # Check if setup was defined previously
    domain.name = domain.setup.node_name


if not len(domain.roles):  # Check if roles were registered previously
    seed_db(SessionLocal())
