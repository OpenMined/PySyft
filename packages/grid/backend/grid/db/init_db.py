# stdlib
import os
import time
from typing import Optional

# third party
from nacl.encoding import HexEncoder
from nacl.signing import SigningKey
from sqlalchemy.orm import Session

# grid absolute
from grid.core.config import settings
from grid.core.node import node


def init_db(db: Session, signing_key: Optional[SigningKey] = None) -> None:
    if not len(node.setup):  # Check if setup was defined previously
        if signing_key is None:
            signing_key = SigningKey.generate()

        service_name = os.getenv("SERVICE_NAME", "")
        # If it's backend service, then create the first account
        if service_name == "backend":
            node.initial_setup(
                signing_key=signing_key,
                first_superuser_name="Jane Doe",
                first_superuser_email=settings.FIRST_SUPERUSER,
                first_superuser_password=settings.FIRST_SUPERUSER_PASSWORD,
                first_superuser_budget=5.55,
                domain_name=settings.DOMAIN_NAME,
            )
            return
        else:
            # If you're not the backend service but still need to init db (celeryworker/backend_stream)
            # You just need to wait it to be created by the backend and check if the setup has been created
            # in the database.
            while not len(node.setup):
                print("waiting for the setup to be created ...")
                time.sleep(1)

    # Uses the Owner root key to update node keys
    setup = node.setup.first()
    if setup.signing_key:
        # the system has a key so use it
        signing_key = SigningKey(setup.signing_key.encode("utf-8"), encoder=HexEncoder)
    else:
        # the system doesn't have a key from older versions
        # so lets get the root user one and use that
        owner = node.users.first(role=node.roles.owner_role.id)
        signing_key = SigningKey(owner.private_key.encode("utf-8"), encoder=HexEncoder)
        node.setup.update(**{"signing_key": owner.private_key})

    type(node).set_keys(node=node, signing_key=signing_key)
