# stdlib
from typing import Optional

# third party
from nacl.signing import SigningKey

# syft absolute
from syft import Domain  # type: ignore
from syft import Network  # type: ignore
from syft.core.node.common.client import Client
from syft.core.node.common.node_table import Base
from syft.core.node.common.node_table.utils import seed_db

# grid absolute
from app.core.config import settings
from app.db.session import get_db_engine
from app.db.session import get_db_session

if settings.NODE_TYPE.lower() == "domain":
    node = Domain("Domain", db_engine=get_db_engine())
elif settings.NODE_TYPE.lower() == "network":
    node = Network("Network", db_engine=get_db_engine())
else:
    raise Exception(
        "Don't know NODE_TYPE "
        + str(settings.NODE_TYPE)
        + ". Please set "
        + "NODE_TYPE to either 'Domain' or 'Network'."
    )

node.loud_print()
Base.metadata.create_all(get_db_engine())

if len(node.setup):  # Check if setup was defined previously
    node.name = node.setup.node_name


if not len(node.roles):  # Check if roles were registered previously
    seed_db(get_db_session())


def get_client(signing_key: Optional[SigningKey] = None) -> Client:
    return node.get_client(signing_key=signing_key)
