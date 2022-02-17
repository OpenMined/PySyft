"""remove group table

Revision ID: 6ebc05b27174
Revises: ada15f85c3d0
Create Date: 2022-01-31 07:11:32.471320

"""
import sqlalchemy as sa

# third party
from alembic import op  # type: ignore

# revision identifiers, used by Alembic.
revision = "6ebc05b27174"
down_revision = "ada15f85c3d0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_table("usergroup")
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
        sa.ForeignKeyConstraint(
            ["group"],
            ["group.id"],
        ),
        sa.ForeignKeyConstraint(
            ["user"],
            ["syft_user.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
