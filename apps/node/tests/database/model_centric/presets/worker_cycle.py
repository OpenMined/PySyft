from random import randint

from src.app.main.model_centric.workers.worker import Worker

from .. import BIG_INT

WORKERS = [
    Worker(
        id=randint(0, BIG_INT),
        ping=randint(0, 100),
        avg_download=randint(0, 100),
        avg_upload=randint(0, 100),
    )
]
