# third party
from nacl.encoding import HexEncoder
from nacl.signing import SigningKey
from sqlalchemy.orm import Session

# syft absolute
from syft.core.node.common.node_service.node_setup.node_setup_messages import (
    CreateInitialSetUpMessage,
)

# grid absolute
from grid.core.config import settings
from grid.core.node import node
from grid.db import base  # noqa: F401


def init_db(db: Session) -> None:
    if not len(node.setup):  # Check if setup was defined previously
        # Build Syft Message
        msg = CreateInitialSetUpMessage(
            address=node.address,
            name="Jane Doe",
            email=settings.FIRST_SUPERUSER,
            password=settings.FIRST_SUPERUSER_PASSWORD,
            domain_name=settings.DOMAIN_NAME,
            budget=55.55,
            reply_to=node.address,
        ).sign(signing_key=node.signing_key)

        # Process syft message
        _ = node.recv_immediate_msg_with_reply(msg=msg).message

    else:  # Uses the Owner root key to update node keys
        owner = node.users.first(role=node.roles.owner_role.id)
        root_key = SigningKey(owner.private_key.encode("utf-8"), encoder=HexEncoder)
        node.signing_key = root_key
        node.verify_key = root_key.verify_key
        node.root_verify_key = root_key.verify_key
