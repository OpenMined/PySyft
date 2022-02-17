"""store write permissions as meta information

Revision ID: ada15f85c3d0
Revises: 6ebc05b27174
Create Date: 2022-01-25 10:30:07.890360

"""
# third party
from alembic import op  # type: ignore
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "ada15f85c3d0"
down_revision = "6ebc05b27174"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "obj_metadata",
        sa.Column("write_permissions", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("obj_metadata", "write_permissions")
