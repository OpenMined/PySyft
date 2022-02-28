"""empty message

Revision ID: 70fcad0b1795
Revises: 6ebc05b27174
Create Date: 2022-02-14 14:40:56.479413

"""
# third party
from alembic import op  # type: ignore
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "70fcad0b1795"
down_revision = "6ebc05b27174"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint("obj_metadata_obj_fkey", "obj_metadata", type_="foreignkey")
    op.drop_constraint(
        "bin_obj_dataset_obj_fkey", "bin_obj_dataset", type_="foreignkey"
    )
    op.drop_table("bin_object")


def downgrade() -> None:
    op.create_table(
        "bin_object",
        sa.Column("id", sa.String(length=256), nullable=False),
        sa.Column("binary", sa.LargeBinary(length=3072), nullable=True),
        sa.Column("obj_name", sa.String(length=3072), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_foreign_key(
        "obj_metadata_obj_fkey",
        "obj_metadata",
        "bin_object",
        ["obj"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "bin_obj_dataset_obj_fkey",
        "bin_obj_dataset",
        "bin_object",
        ["obj"],
        ["id"],
        ondelete="CASCADE",
    )
