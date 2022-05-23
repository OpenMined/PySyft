"""empty message

Revision ID: 7ae5bd4da7ef
Revises: f712122fe780
Create Date: 2022-05-17 07:52:14.126268

"""
# third party
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "7ae5bd4da7ef"
down_revision = "f712122fe780"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("association_request", sa.Column("node_id", sa.String(), default=""))
    op.drop_column("association_request", "source")
    op.drop_column("association_request", "target")


def downgrade() -> None:
    op.drop_column("association_request", "node_id")
    op.add_column("association_request", sa.Column("source", sa.String(), default=""))
    op.add_column("association_request", sa.Column("target", sa.String(), default=""))
