# third party
from sqlalchemy import create_engine
from sqlalchemy.engine.base import Engine
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker

# grid absolute
from app.core.config import settings


def get_db_engine(db_uri: str = str(settings.SQLALCHEMY_DATABASE_URI)) -> Engine:
    if db_uri.startswith("sqlite://"):
        return create_engine(db_uri, echo=False)
    return create_engine(db_uri, pool_pre_ping=True)


def get_db_session(db_uri: str = str(settings.SQLALCHEMY_DATABASE_URI)) -> Session:
    engine = get_db_engine(db_uri=db_uri)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()
