# Import all the models, so that Base has them before being imported by alembic
from syft.core.node.common.node_table import Base  # noqa
