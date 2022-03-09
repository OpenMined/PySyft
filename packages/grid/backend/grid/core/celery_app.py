# future
from __future__ import annotations

# third party
from celery import Celery

# relative
from . import celery_config
from . import celery_serde  # noqa: 401

# backend is required to persist tasks
celery_app = Celery("worker", broker="amqp://guest@queue//")
celery_app.config_from_object(celery_config)
