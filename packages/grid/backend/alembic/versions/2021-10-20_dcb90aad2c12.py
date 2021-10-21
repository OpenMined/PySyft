"""empty message

Revision ID: dcb90aad2c12
Revises: 239c5dd652ba
Create Date: 2021-10-20 16:56:21.015916

"""
import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "dcb90aad2c12"
down_revision = "239c5dd652ba"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "node",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("node_id", sa.String(length=255)),
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
