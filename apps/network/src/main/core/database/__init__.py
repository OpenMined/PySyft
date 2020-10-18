import os
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from sqlalchemy_mixins import AllFeaturesMixin
from sqlalchemy_utils.functions import database_exists

db = SQLAlchemy()


class BaseModel(db.Model, AllFeaturesMixin):
    __abstract__ = True
    pass


from .roles.roles import Role, create_role
from .users.user import User, create_user
from .utils import model_to_json


def set_database_config(app, test_config=None, verbose=False):
    """Set configs to use SQL Alchemy library.

    Args:
        app: Flask application.
        test_config : Dictionary containing SQLAlchemy configs for test purposes.
        verbose : Level of flask application verbosity.
    Returns:
        app: Flask application.
    Raises:
        RuntimeError : If DATABASE_URL or test_config didn't initialize, RuntimeError exception will be raised.
    """
    db_url = os.environ.get("DATABASE_URL")
    migrate = Migrate(app, db)
    if test_config is None:
        if db_url:
            app.config.from_mapping(
                SQLALCHEMY_DATABASE_URI=db_url, SQLALCHEMY_TRACK_MODIFICATIONS=False
            )
        else:
            raise RuntimeError(
                "Invalid database address : Set DATABASE_URL environment var or add test_config parameter at create_app method."
            )
    else:
        app.config["SQLALCHEMY_DATABASE_URI"] = test_config["SQLALCHEMY_DATABASE_URI"]
        app.config["TESTING"] = (
            test_config["TESTING"] if test_config.get("TESTING") else True
        )
        app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = (
            test_config["SQLALCHEMY_TRACK_MODIFICATIONS"]
            if test_config.get("SQLALCHEMY_TRACK_MODIFICATIONS")
            else False
        )
    app.config["VERBOSE"] = verbose
    db.init_app(app)


def seed_db():
    """Adds Administrator and Owner Roles to database."""
    global db
    new_user = Role(
        name="Administrator",
        can_edit_settings=False,
        can_create_users=False,
        can_edit_roles=False,
        can_manage_nodes=False,
    )
    db.session.add(new_user)
    new_user = Role(
        name="Owner",
        can_edit_settings=True,
        can_create_users=True,
        can_edit_roles=True,
        can_manage_nodes=True,
    )
    db.session.add(new_user)

    db.session.commit()
