"""empty message

Revision ID: 741a16c345bf
Revises: 169d1a52c0c6
Create Date: 2022-06-24 03:07:45.613824

"""
# third party
from alembic import op  # type: ignore
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "741a16c345bf"
down_revision = "169d1a52c0c6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "node_route",
        sa.Column("protocol", sa.String(length=255), nullable=False, default="http"),
    )
    op.add_column(
        "node_route",
        sa.Column("port", sa.Integer(), nullable=False, default=80),
    )
    op.add_column(
        "node_route",
        sa.Column("private", sa.Boolean(), nullable=False, default=False),
    )


def downgrade() -> None:
    op.drop_column("node_route", "protocol")
    op.drop_column("node_route", "port")
    op.drop_column("node_route", "private")
