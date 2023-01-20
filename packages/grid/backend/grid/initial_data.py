# stdlib
import logging
import os

# grid absolute
from grid.db.init_db import init_db
from grid.db.init_db import load_db
from grid.db.session import get_db_session

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init() -> None:
    db = get_db_session()
    service_name = os.getenv("SERVICE_NAME", "")
    if service_name == "backend":
        init_db(db)
    else:
        load_db(db)


def main() -> None:
    logger.info("Creating initial data")
    init()
    logger.info("Initial data created")


if __name__ == "__main__":
    main()
