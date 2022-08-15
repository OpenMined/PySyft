"""empty message

Revision ID: b70cd0f15740
Revises: 741a16c345bf
Create Date: 2022-08-15 03:52:45.076651

"""
# third party
from alembic import op  # type: ignore
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "b70cd0f15740"
down_revision = "741a16c345bf"
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
