""" A set of useful methods used by the syft.core.node.common.node_table submodule. """
# stdlib
from typing import Any
from typing import Dict
from typing import Tuple as TypeTuple

# third party
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker

# relative
from . import Base
from .roles import Role

# attribute names representing a date owned by the PyGrid's database schemas.
datetime_cols = ["date", "created_at", "destroyed_at", "deployed_on", "updated_on"]


def model_to_json(model: Base) -> Dict[str, Any]:
    """
    Returns a JSON representation of an SQLAlchemy-backed object.

    Args:
        model: SQLAlchemy-backed object to be represented as a JSON data structure.
    Returns:
        Dict: Python dictionary representing the SQLAlchemy object.
    """
    json = {}
    for col in model.__mapper__.attrs.keys():  # type: ignore
        if col != "hashed_password" and col != "salt":
            if col in datetime_cols:
                # Cast datetime object to string
                json[col] = str(getattr(model, col))
            else:
                json[col] = getattr(model, col)
    return json


def seed_db(db: Session) -> None:
    """Creates the standard PyGrid roles and seeds it into the database.

    Args:
        db (Session): Database Session object used to handle database operations.
    """
    new_role = Role(
        name="Data Scientist",
        can_make_data_requests=True,
        can_triage_data_requests=False,
        can_manage_privacy_budget=False,
        can_create_users=False,
        can_manage_users=False,
        can_edit_roles=False,
        can_manage_infrastructure=False,
        can_upload_data=False,
        can_upload_legal_document=False,
        can_edit_domain_settings=False,
    )
    db.add(new_role)

    new_role = Role(
        name="Compliance Officer",
        can_make_data_requests=False,
        can_triage_data_requests=True,
        can_manage_privacy_budget=False,
        can_create_users=False,
        can_manage_users=True,
        can_edit_roles=False,
        can_manage_infrastructure=False,
        can_upload_data=False,
        can_upload_legal_document=False,
        can_edit_domain_settings=False,
    )
    db.add(new_role)

    new_role = Role(
        name="Administrator",
        can_make_data_requests=True,
        can_triage_data_requests=True,
        can_manage_privacy_budget=True,
        can_create_users=True,
        can_manage_users=True,
        can_edit_roles=True,
        can_manage_infrastructure=True,
        can_upload_data=True,
        can_upload_legal_document=True,
        can_edit_domain_settings=True,
    )
    db.add(new_role)

    new_role = Role(
        name="Owner",
        can_make_data_requests=True,
        can_triage_data_requests=True,
        can_manage_privacy_budget=True,
        can_create_users=True,
        can_manage_users=True,
        can_edit_roles=True,
        can_manage_infrastructure=True,
        can_upload_data=True,
        can_upload_legal_document=True,
        can_edit_domain_settings=True,
    )
    db.add(new_role)
    db.commit()


def create_memory_db_engine() -> TypeTuple[Engine, sessionmaker]:
    """Creates the database tables in memory instead of connecting with a real database server.

    Returns:
        Tuple: Tuple containing both database engine and session maker.
    """
    db_engine = create_engine("sqlite://", echo=False)
    Base.metadata.create_all(db_engine)  # type: ignore
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)
    seed_db(SessionLocal())
    return db_engine, SessionLocal
