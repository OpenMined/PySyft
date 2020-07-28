import pytest
import sys

from . import BIG_INT
from .presets.worker import worker_metrics

from random import randint
from src.app.main.sfl.workers.worker import Worker

sys.path.append(".")


@pytest.mark.parametrize("ping, avg_download, avg_upload", worker_metrics)
def test_create_worker_object(ping, avg_download, avg_upload, database):
    worker = Worker(
        id=randint(0, BIG_INT),
        ping=ping,
        avg_download=avg_download,
        avg_upload=avg_upload,
    )
    database.session.add(worker)
    database.session.commit()
