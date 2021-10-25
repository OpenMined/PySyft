"""Update requests table

Revision ID: 100642749d64
Revises: 1bdbdf7f26ce
Create Date: 2021-10-06 09:22:40.298132

"""
# third party
from alembic import op  # type: ignore
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "100642749d64"
down_revision = "1bdbdf7f26ce"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("request", sa.Column("user_email", sa.String(255), default=""))
    op.add_column("request", sa.Column("user_role", sa.String(255), default=""))
    op.add_column("request", sa.Column("institution", sa.String(255), default=""))
    op.add_column("request", sa.Column("user_budget", sa.Float(), default=0.0))
    op.add_column("request", sa.Column("website", sa.String(255), default=""))
    op.add_column("request", sa.Column("reviewer_name", sa.String(255), default=""))
    op.add_column("request", sa.Column("reviewer_role", sa.String(255), default=""))
    op.add_column("request", sa.Column("reviewer_comment", sa.String(255), default=""))
    op.add_column("request", sa.Column("requested_budget", sa.Float(), default=0.0))
    op.add_column("request", sa.Column("current_budget", sa.Float(), default=0.0))
    op.add_column("request", sa.Column("updated_on", sa.DateTime(), nullable=True))
    op.add_column("syft_application", sa.Column("budget", sa.Float(), default=0.0))


def downgrade() -> None:
    pass
