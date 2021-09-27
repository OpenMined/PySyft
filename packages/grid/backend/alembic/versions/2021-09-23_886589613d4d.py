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
        sa.ForeignKeyConstraint(
            ["daa_pdf"],
            ["daa_pdf.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    pass


def downgrade() -> None:
    pass
