# grid absolute
from grid.core.config import settings
from grid.core.node import node


def init_db() -> None:
    """This function is executed by the backend service and it checks if setup table
    was already initialized. If not, then we create a initial setup with the first account
    (Domain Owner). Otherwise, we just load the setup from the database directly.
    Parameters:
        signing_key: Optional signing key is case we want to use an specific key to use some specific
        key to be our node root key.
    """

    node.initial_setup(
        signing_key=node.signing_key,
        first_superuser_name="Jane Doe",
        first_superuser_email=settings.FIRST_SUPERUSER,
        first_superuser_password=settings.FIRST_SUPERUSER_PASSWORD,
        first_superuser_budget=5.55,
        domain_name=settings.DOMAIN_NAME,
    )


if __name__ == "__main__":
    init_db()
