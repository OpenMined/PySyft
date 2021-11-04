# future
from __future__ import annotations

# stdlib
from typing import Any

# third party
from celery import Celery
from google.protobuf.reflection import GeneratedProtocolMessageType
from kombu import serialization

# syft absolute
import syft as sy
from syft.logger import error


def loads(data: bytes) -> Any:
    # original payload might have nested bytes in the args
    org_payload = sy.deserialize(data, from_bytes=True).upcast()
    # original payload is found at org_payload[0][0]
    if (
        len(org_payload) > 0
        and len(org_payload[0]) > 0
        and isinstance(org_payload[0][0], bytes)
    ):
        try:
            nested_data = org_payload[0][0]
            org_obj = sy.deserialize(nested_data, from_bytes=True)
            org_payload[0][0] = org_obj
        except Exception as e:
            error(f"Unable to deserialize nested payload. {e}")
            raise e

    return org_payload


def dumps(obj: Any) -> bytes:
    # this is usually a Tuple of args where the first one is what we send to the task
    # but it can also get other arbitrary data which we need to serde
    # since we might get bytes directly from the web endpoint we can avoid double
    # unserializing it by keeping it inside the nested args list org_payload[0][0]
    return sy.serialize(obj, to_bytes=True)


serialization.register(
    "syft",
    dumps,
    loads,
    content_type="application/syft",
    content_encoding="binary",
)


# backend is required to persist tasks
celery_app = Celery(
    "worker",
    broker="amqp://guest@queue//",
)
# celery_app.conf.result_backend = "db+sqlite:///results.db"
# celery_app.conf.result_backend = "file:///tmp/results"
# celery_app.conf.result_backend = "postgresql://postgres:changethis@docker-host:5432"
# celery_app.conf.result_backend = "amqp://guest@queue//"
celery_app.conf.result_backend = "rpc://"
celery_app.conf.result_persistent = True
celery_app.conf.task_routes = {
    "grid.worker.msg_without_reply": "main-queue",
}
celery_app.worker_prefetch_multiplier = 1
celery_app.conf.accept_content = ["application/syft"]
celery_app.conf.task_serializer = "syft"
celery_app.conf.result_serializer = "syft"
