from random import randint

from .. import BIG_INT

from src.app.main.sfl.workers.worker import Worker

WORKERS = [
    Worker(
        id=randint(0, BIG_INT),
        ping=randint(0, 100),
        avg_download=randint(0, 100),
        avg_upload=randint(0, 100),
    ),
]
