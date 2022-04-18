"""add a boolean flag for proxy dataset in ObjectMetadata

Revision ID: f712122fe780
Revises: 70fcad0b1795
Create Date: 2022-03-31 07:20:49.411961

"""
# third party
from alembic import op  # type: ignore
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "f712122fe780"
down_revision = "70fcad0b1795"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "obj_metadata", sa.Column("is_proxy_dataset", sa.Boolean(), default=False)
    )


def downgrade() -> None:
    op.drop_column("obj_metadata", "is_proxy_dataset")
