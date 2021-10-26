"""empty message

Revision ID: 07fb4c284883
Revises: 100642749d64
Create Date: 2021-10-26 10:41:09.286331

"""
import sqlalchemy as sa

# third party
from alembic import op  # type: ignore

# revision identifiers, used by Alembic.
revision = "07fb4c284883"
down_revision = "100642749d64"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "node",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("node_uid", sa.String(length=255)),
        sa.Column("node_name", sa.String(length=255)),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "node_route",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("node_id", sa.Integer()),
        sa.Column("host_or_ip", sa.String(length=255)),
        sa.Column("is_vpn", sa.Boolean(), default=False),
        sa.ForeignKeyConstraint(["node_id"], ["node.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.drop_table("association")


def downgrade() -> None:
    op.drop_table("node")
    op.drop_table("node_route")
    op.create_table(
        "association",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("date", sa.DateTime(), nullable=True),
        sa.Column("name", sa.String(length=255), nullable=True),
        sa.Column("address", sa.String(length=255), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
