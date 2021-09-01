# stdlib
import logging

# grid absolute
from app.db.init_db import init_db
from app.db.session import get_db_session

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init() -> None:
    db = get_db_session()
    init_db(db)


def main() -> None:
    logger.info("Creating initial data")
    init()
    logger.info("Initial data created")


if __name__ == "__main__":
    main()
