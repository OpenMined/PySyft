"""ADD keep_connected column

Revision ID: a8ffb9432e29
Revises: f712122fe780
Create Date: 2022-05-08 20:07:54.218896

"""
# third party
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "a8ffb9432e29"
down_revision = "f712122fe780"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("node", sa.Column("keep_connected", sa.Boolean(), default=True))
    op.add_column("node_route", sa.Column("vpn_endpoint", sa.String(255), default=""))
    op.add_column("node_route", sa.Column("vpn_key", sa.String(255), default=""))


def downgrade() -> None:
    op.drop_column("node", "keep_connected")
    op.drop_column("node_route", "vpn_endpoint")
    op.drop_column("node_route", "vpn_key")
