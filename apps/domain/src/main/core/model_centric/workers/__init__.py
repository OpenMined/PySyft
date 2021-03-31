from .worker_manager import WorkerManager
from ...database import db

worker_manager = WorkerManager(db)
