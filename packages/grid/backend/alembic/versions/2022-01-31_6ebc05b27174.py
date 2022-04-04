"""remove group table

Revision ID: 6ebc05b27174
Revises: ada15f85c3d0
Create Date: 2022-01-31 07:11:32.471320

"""
# third party
from alembic import op  # type: ignore
import sqlalchemy as sa
from sqlalchemy.engine.reflection import Inspector

# revision identifiers, used by Alembic.
revision = "6ebc05b27174"
down_revision = "ada15f85c3d0"
branch_labels = None
depends_on = None


def get_table_names() -> list:
    conn = op.get_bind()
    inspector = Inspector.from_engine(conn)
    tables = inspector.get_table_names()
    return tables


def upgrade() -> None:
    table_names = get_table_names()
    if "usergroup" in table_names:
        op.drop_table("usergroup")

    if "group" in table_names:
        op.drop_table("group")


def downgrade() -> None:
    op.create_table(
        "group",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(length=255), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "usergroup",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user", sa.Integer(), nullable=True),
        sa.Column("group", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["group"], ["group.id"]),
        sa.ForeignKeyConstraint(["user"], ["syft_user.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
