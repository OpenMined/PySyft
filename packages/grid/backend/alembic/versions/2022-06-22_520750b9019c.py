"""empty message

Revision ID: 520750b9019c
Revises: f712122fe780
Create Date: 2022-06-22 01:42:48.998662

"""
# third party
from alembic import op  # type: ignore
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "520750b9019c"
down_revision = "f712122fe780"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "node",
        sa.Column("node_type", sa.String(length=255), nullable=True),
    )
    op.add_column(
        "node",
        sa.Column("verify_key", sa.String(length=2048), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("node", "node_type")
    op.drop_column("node", "verify_key")
