# third party
from nacl.encoding import HexEncoder
from nacl.signing import SigningKey
from sqlalchemy.orm import Session

# grid absolute
from grid.core.config import settings
from grid.core.node import node


def init_db(db: Session) -> None:
    if not len(node.setup):  # Check if setup was defined previously
        node.initial_setup(
            first_superuser_name="Jane Doe",
            first_superuser_email=settings.FIRST_SUPERUSER,
            first_superuser_password=settings.FIRST_SUPERUSER_PASSWORD,
            first_superuser_budget=5.55,
            domain_name=settings.DOMAIN_NAME,
        )
    else:  # Uses the Owner root key to update node keys
        owner = node.users.first(role=node.roles.owner_role.id)
        root_key = SigningKey(owner.private_key.encode("utf-8"), encoder=HexEncoder)
        node.signing_key = root_key
        node.verify_key = root_key.verify_key
        node.root_verify_key = root_key.verify_key
