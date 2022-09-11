# stdlib
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

        node.initial_setup(
            signing_key=signing_key,
            first_superuser_name="Jane Doe",
            first_superuser_email=settings.FIRST_SUPERUSER,
            first_superuser_password=settings.FIRST_SUPERUSER_PASSWORD,
            first_superuser_budget=5.55,
            domain_name=settings.DOMAIN_NAME,
        )
    else:  # Uses the Owner root key to update node keys
        setup = node.setup.first()
        if setup.signing_key:
            # the system has a key so use it
            signing_key = SigningKey(
                setup.signing_key.encode("utf-8"), encoder=HexEncoder
            )
        else:
            # the system doesn't have a key from older versions
            # so lets get the root user one and use that
            owner = node.users.first(role=node.roles.owner_role.id)
            signing_key = SigningKey(
                owner.private_key.encode("utf-8"), encoder=HexEncoder
            )
            node.setup.update_config(**{"signing_key": owner.private_key})

        type(node).set_keys(node=node, signing_key=signing_key)
