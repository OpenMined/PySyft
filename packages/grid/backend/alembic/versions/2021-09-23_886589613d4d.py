"""empty message

Revision ID: 886589613d4d
Revises: 916812f40fb4
Create Date: 2021-09-23 07:45:53.472208

"""
# third party
from alembic import op  # type: ignore
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "886589613d4d"
down_revision = "916812f40fb4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "daa_pdf",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("binary", sa.LargeBinary(length=3072), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "syft_application",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("email", sa.String(length=255), nullable=True),
        sa.Column("name", sa.String(length=255), nullable=True),
        sa.Column("hashed_password", sa.String(length=512), nullable=True),
        sa.Column("salt", sa.String(length=255), nullable=True),
        sa.Column("daa_pdf", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(length=255), nullable=True),
        sa.Column("website", sa.String(2048), default=""),
        sa.Column("institution", sa.String(2048), default=""),
        sa.Column("added_by", sa.String(2048), default=""),
        sa.ForeignKeyConstraint(["daa_pdf"], ["daa_pdf.id"]),
        sa.PrimaryKeyConstraint("id"),
    )

    op.add_column("syft_user", sa.Column("website", sa.String(2048), default=""))
    op.add_column("syft_user", sa.Column("institution", sa.String(2048), default=""))
    op.add_column("syft_user", sa.Column("added_by", sa.String(2048), default=""))
    op.add_column("syft_user", sa.Column("daa_pdf", sa.Integer(), nullable=True))
    op.add_column("syft_user", sa.Column("created_at", sa.DateTime(), nullable=True))
    op.add_column("syft_user", sa.Column("allocated_budget", sa.Float(), default=0.0))

    op.add_column("setup", sa.Column("tags", sa.String(2048), default="[]"))
    op.add_column("setup", sa.Column("deployed_on", sa.DateTime(), nullable=True))
    pass


def downgrade() -> None:
    pass
