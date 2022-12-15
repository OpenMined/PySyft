# stdlib
import logging
import os

# grid absolute
from grid.db.init_db import init_db
from grid.db.init_db import load_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init() -> None:
    service_name = os.getenv("SERVICE_NAME", "")
    if service_name == "backend":
        init_db()
    else:
        load_db()


def main() -> None:
    logger.info("Creating initial data")
    init()
    logger.info("Initial data created")


if __name__ == "__main__":
    main()
