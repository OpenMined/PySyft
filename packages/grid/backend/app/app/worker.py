# third party
from raven import Client

# syft absolute
from syft import deserialize
from syft.core.common import UID
from syft.core.store.storeable_object import StorableObject
from syft.lib.python import Int

# grid absolute
from app.core.celery_app import celery_app
from app.core.config import settings
from app.core.node import node

client_sentry = Client(settings.SENTRY_DSN)


@celery_app.task(acks_late=True)
def add_num(arg_bytes_str: str) -> None:
    # use latin-1 instead of utf-8 because our bytes might not be an even number
    arg_bytes = bytes(arg_bytes_str, "latin-1")
    args = deserialize(arg_bytes, from_bytes=True)

    num = args["num"]
    id_at_location = args["id_at_location"]
    verify_key = args["verify_key"]

    # stdlib
    import random

    rand = round(random.random() * 10)
    new_num = num + rand
    print(f"added {rand} to {num} == {new_num}")
    result = StorableObject(
        id=id_at_location,
        data=new_num,
        tags=[],
        description="",
        search_permissions={verify_key: None},
        read_permissions={verify_key: None},
    )

    # stdlib
    import time

    time.sleep(10)
    node.store[id_at_location] = result
