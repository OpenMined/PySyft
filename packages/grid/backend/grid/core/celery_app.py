# third party
from celery import Celery

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
