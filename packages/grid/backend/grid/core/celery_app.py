# third party
from celery import Celery

# relative
from . import celery_config

# backend is required to persist tasks
celery_app = Celery(
    "worker",
    broker="amqp://guest@queue//",
)
celery_app.config_from_object(celery_config)
celery_app.conf.task_routes = {
    "grid.worker.msg_without_reply": "main-queue",
    "delivery_mode": "transient",
}
