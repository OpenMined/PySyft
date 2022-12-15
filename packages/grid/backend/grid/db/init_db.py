# stdlib
import time
from typing import Optional

# third party
from nacl.encoding import HexEncoder
from nacl.signing import SigningKey

# grid absolute
from grid.core.config import settings
from grid.core.node import node


def load_db() -> None:
    """This function is executed by the node services (backend-stream/celery workers)
    and also for backend service when we already have a setup defined. This function will
    just load setup table initialized previously and update domain Object root key.

    Parameters:
    """
    # Wait until setup is created by backend service
    while not len(node.setup):
        time.sleep(1)

    # Get Setup table
    setup = node.setup.first()
    # If Node's Signing key  was defined in setup table, the we can use it
    # to our domain as well.
    if setup.signing_key:
        signing_key = SigningKey(setup.signing_key.encode("utf-8"), encoder=HexEncoder)
    else:  # If not, then we use the domain owner signing key.
        owner = node.users.first(role=node.roles.owner_role.id)
        signing_key = SigningKey(owner.private_key.encode("utf-8"), encoder=HexEncoder)
        node.setup.update_config(**{"signing_key": owner.private_key})

    # Update Node object signing/verify keys.
    type(node).set_keys(node=node, signing_key=signing_key)


def init_db(signing_key: Optional[SigningKey] = None) -> None:
    """This function is executed by the backend service and it checks if setup table
    was already initialized. If not, then we create a initial setup with the first account
    (Domain Owner). Otherwise, we just load the setup from the database directly.

    Parameters:
        signing_key: Optional signing key is case we want to use an specific key to use some specific
        key to be our node root key.
    """
    if not len(node.setup):  # Check if setup was defined previously
        if signing_key is None:
            signing_key = SigningKey.generate()

        node.initial_setup(
            signing_key=signing_key,
            first_superuser_name="Jane Doe",
            first_superuser_email=settings.FIRST_SUPERUSER,
            first_superuser_password=settings.FIRST_SUPERUSER_PASSWORD,
            first_superuser_budget=5.55,
            domain_name=settings.DOMAIN_NAME,
        )
    else:
        load_db()
