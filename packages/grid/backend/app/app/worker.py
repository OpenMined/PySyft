# third party
from raven import Client
from syft.core.common.message import SignedImmediateSyftMessageWithoutReply

# grid absolute
from app.core.celery_app import celery_app
from app.core.config import settings
from app.core.node import node

# syft absolute
from syft import deserialize

client_sentry = Client(settings.SENTRY_DSN)


@celery_app.task(acks_late=True)
def msg_without_reply(msg_bytes_str: str) -> None:
    # use latin-1 instead of utf-8 because our bytes might not be an even number
    msg_bytes = bytes(msg_bytes_str, "latin-1")
    print("string message", msg_bytes)
    obj_msg = deserialize(blob=msg_bytes, from_bytes=True)
    print("Message", obj_msg)
    if isinstance(obj_msg, SignedImmediateSyftMessageWithoutReply):
        node.recv_immediate_msg_without_reply(msg=obj_msg)
    else:
        raise Exception(
            f"This worker can only handle SignedImmediateSyftMessageWithoutReply. {msg_bytes_str}"
        )
