"""Update roles

Revision ID: 239c5dd652ba
Revises: bb642928e749
Create Date: 2021-09-27 04:38:23.860642

"""
# third party
from alembic import op  # type: ignore
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "239c5dd652ba"
down_revision = "bb642928e749"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_column("role", "can_triage_requests")
    op.drop_column("role", "can_edit_settings")
    op.drop_column("role", "can_create_groups")

    op.add_column(
        "role", sa.Column("can_make_data_requests", sa.Boolean(), default=False)
    )
    op.add_column(
        "role", sa.Column("can_triage_data_requests", sa.Boolean(), default=False)
    )
    op.add_column(
        "role", sa.Column("can_manage_privacy_budget", sa.Boolean(), default=False)
    )
    op.add_column("role", sa.Column("can_manage_users", sa.Boolean(), default=False))
    op.add_column(
        "role", sa.Column("can_upload_legal_document", sa.Boolean(), default=False)
    )
    op.add_column(
        "role", sa.Column("can_edit_domain_settings", sa.Boolean(), default=False)
    )


def downgrade() -> None:
    pass
