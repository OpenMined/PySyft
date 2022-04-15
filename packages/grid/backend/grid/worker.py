# stdlib
from typing import Any

# syft absolute
from syft.core.common.message import SignedImmediateSyftMessageWithoutReply

# grid absolute
from grid.core.celery_app import celery_app
from grid.core.config import settings  # noqa: F401
from grid.core.node import node
from grid.periodic_tasks import cleanup_incomplete_uploads_from_blob_store

# TODO : Should be modified to use exponential backoff (for efficiency)
# Initially we have set 0.1 as the retry time.
# We have set max retries =(1200) 120 seconds


@celery_app.task(bind=True, acks_late=True)
def msg_without_reply(self, obj_msg: Any) -> None:  # type: ignore
    if isinstance(obj_msg, SignedImmediateSyftMessageWithoutReply):
        try:
            node.recv_immediate_msg_without_reply(msg=obj_msg)
        except Exception as e:
            raise e
    else:
        raise Exception(
            f"This worker can only handle SignedImmediateSyftMessageWithoutReply. {obj_msg}"
        )


@celery_app.on_after_configure.connect
def setup_periodic_task(sender, **kwargs) -> None:  # type: ignore
    celery_app.add_periodic_task(
        3600,  # Run every hour
        cleanup_incomplete_uploads_from_blob_store.s(),
        name="Clean incomplete uploads in Seaweed",
        queue="main-queue",
        options={"queue": "main-queue"},
    )
