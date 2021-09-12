"""Add description, daa and  contact columns

Revision ID: 5796f6ceb314
Revises: bb642928e749
Create Date: 2021-09-10 08:00:06.883470

"""
# third party
from alembic import op  # type: ignore
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "5796f6ceb314"
down_revision = "bb642928e749"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("setup", sa.Column("description", sa.String(255), default=""))
    op.add_column("setup", sa.Column("contact", sa.String(255), default=""))
    op.add_column("setup", sa.Column("daa", sa.Boolean(), default=False))
    pass


def downgrade() -> None:
    pass
