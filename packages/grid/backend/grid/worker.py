# third party
from raven import Client

# syft absolute
from syft import deserialize  # type: ignore
from syft.core.common.message import SignedImmediateSyftMessageWithoutReply
from syft.core.node.common.action.smpc_action_message import ObjectNotInStore

# grid absolute
from grid.core.celery_app import celery_app
from grid.core.config import settings
from grid.core.node import node

client_sentry = Client(settings.SENTRY_DSN)

# TODO : Should be modified to use exponential backoff (for efficiency)
# Initially we have set 0.1 as the retry time.
# We have set max retries =(1200) 120 seconds


@celery_app.task(bind=True, max_retries=1200, acks_late=True)
def msg_without_reply(self, msg_bytes_str: str) -> None:  # type: ignore
    # use latin-1 instead of utf-8 because our bytes might not be an even number
    msg_bytes = bytes(msg_bytes_str, "latin-1")
    obj_msg = deserialize(blob=msg_bytes, from_bytes=True)
    if isinstance(obj_msg, SignedImmediateSyftMessageWithoutReply):
        try:
            node.recv_immediate_msg_without_reply(msg=obj_msg)
        except ObjectNotInStore as exc:
            raise self.retry(exc=exc, countdown=0.1)
    else:
        raise Exception(
            f"This worker can only handle SignedImmediateSyftMessageWithoutReply. {msg_bytes_str}"
        )
