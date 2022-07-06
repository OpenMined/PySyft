"""empty message

Revision ID: 169d1a52c0c6
Revises: 520750b9019c
Create Date: 2022-06-23 03:38:16.036047

"""
# third party
from alembic import op  # type: ignore
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "169d1a52c0c6"
down_revision = "520750b9019c"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "setup",
        sa.Column("signing_key", sa.String(length=2048), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("node", "signing_key")
