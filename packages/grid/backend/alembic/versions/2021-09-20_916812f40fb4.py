"""ADD daa_document column at setup table

Revision ID: 916812f40fb4
Revises: 5796f6ceb314
Create Date: 2021-09-20 01:07:37.239186

"""
# third party
from alembic import op  # type: ignore
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "916812f40fb4"
down_revision = "5796f6ceb314"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("setup", sa.Column("daa_document", sa.String(255), default=""))
    pass


def downgrade() -> None:
    pass
