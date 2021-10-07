# third party
from sqlalchemy.orm import Session

# grid absolute
from grid.core.config import settings
from grid.core.node import node


def init_db(db: Session) -> None:

    node.initial_setup(
        first_superuser_name="Jane Doe",
        first_superuser_email=settings.FIRST_SUPERUSER,
        first_superuser_password=settings.FIRST_SUPERUSER_PASSWORD,
        first_superuser_budget=5.55,
        domain_name=settings.DOMAIN_NAME,
    )
