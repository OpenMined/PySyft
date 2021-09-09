# third party
from raven import Client

# syft absolute
from syft import deserialize  # type: ignore
from syft.core.common.message import SignedImmediateSyftMessageWithoutReply

# grid absolute
from grid.core.celery_app import celery_app
from grid.core.config import settings
from grid.core.node import node

client_sentry = Client(settings.SENTRY_DSN)


@celery_app.task(acks_late=True)
def msg_without_reply(msg_bytes_str: str) -> None:
    # use latin-1 instead of utf-8 because our bytes might not be an even number
    msg_bytes = bytes(msg_bytes_str, "latin-1")
    obj_msg = deserialize(blob=msg_bytes, from_bytes=True)
    if isinstance(obj_msg, SignedImmediateSyftMessageWithoutReply):
        node.recv_immediate_msg_without_reply(msg=obj_msg)
    else:
        raise Exception(
            f"This worker can only handle SignedImmediateSyftMessageWithoutReply. {msg_bytes_str}"
        )
