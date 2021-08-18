# third party
from sqlalchemy.orm import Session

# syft absolute
from syft.core.node.common.node_service.node_setup.node_setup_messages import (
    CreateInitialSetUpMessage,
)

# grid absolute
from app import crud
from app import schemas
from app.core.config import settings
from app.core.node import node
from app.db import base  # noqa: F401

# make sure all SQL Alchemy models are imported (app.db.base) before initializing DB
# otherwise, SQL Alchemy might fail to initialize relationships properly
# for more details: https://github.com/tiangolo/full-stack-fastapi-postgresql/issues/28


def init_db(db: Session) -> None:
    # Tables should be created with Alembic migrations
    # But if you don't want to use migrations, create
    # the tables un-commenting the next line
    # Base.metadata.create_all(bind=engine)z§

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

    user = crud.user.get_by_email(db, email=settings.FIRST_SUPERUSER)
    if not user:
        user_in = schemas.UserCreate(
            email=settings.FIRST_SUPERUSER,
            password=settings.FIRST_SUPERUSER_PASSWORD,
            is_superuser=True,
        )
        user = crud.user.create(db, obj_in=user_in)  # noqa: F841
