"""
 A set of useful functions to handle with SQLAlchemy database session.
"""
# third party
from sqlalchemy import create_engine
from sqlalchemy.engine.base import Engine
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker

# grid absolute
from grid.core.config import settings
from grid.db.base import Base


def get_db_engine(db_uri: str = str(settings.SQLALCHEMY_DATABASE_URI)) -> Engine:
    """
    Creates a new SQLAlchemy database engine object using memory(sqlite)
    or connecting to a real postgresql database.

    :param db_uri: database's address.
    :return: SQLALchemy database engine.
    :rtype: Engine
    """
    if db_uri.startswith("sqlite://"):

        db_engine = create_engine(db_uri, echo=False)
        # TODO change to use alembic properly with the sqlite memory store:
        # https://stackoverflow.com/questions/31406359/use-alembic-to-upgrade-in-memory-sqlite3-database
        Base.metadata.create_all(db_engine)
    else:
        db_engine = create_engine(
            db_uri, pool_pre_ping=True, pool_size=1000, max_overflow=50
        )
    # Base.metadata.create_all(db_engine)
    return db_engine


def get_db_session(db_uri: str = str(settings.SQLALCHEMY_DATABASE_URI)) -> Session:
    """
    Opens a new database session.

    :param db_uri: database's address.
    :return: opened database session.
    :rtype: Session.
    """
    engine = get_db_engine(db_uri=db_uri)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()
