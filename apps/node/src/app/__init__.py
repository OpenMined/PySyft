import logging
import os

from .util import mask_payload_fast
from flask import Flask
from flask_cors import CORS
from flask_executor import Executor
from flask_migrate import Migrate
from flask_sockets import Sockets
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy_mixins import AllFeaturesMixin
from sqlalchemy_utils.functions import database_exists
from geventwebsocket.websocket import Header

# Monkey patch geventwebsocket.websocket.Header.mask_payload() and
# geventwebsocket.websocket.Header.unmask_payload(), for efficiency
Header.mask_payload = mask_payload_fast
Header.unmask_payload = mask_payload_fast

from .version import __version__

# Default secret key used only for testing / development
DEFAULT_SECRET_KEY = "justasecretkeythatishouldputhere"

db = SQLAlchemy()
executor = Executor()
logging.getLogger().setLevel(logging.INFO)


class BaseModel(db.Model, AllFeaturesMixin):
    __abstract__ = True
    pass


# Tables must be created after db has been created
from .main.users import Role


def set_database_config(app, test_config=None, verbose=False):
    """Set configs to use SQL Alchemy library.

    Args:
        app: Flask application.
        test_config : Dictionary containing SQLAlchemy configs for test purposes.
        verbose : Level of flask application verbosity.
    Returns:
        app: Flask application.
    Raises:
        RuntimeError : If DATABASE_URL or test_config didn't initialized, RuntimeError exception will be raised.
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
    global db

    new_role = Role(
        name="User",
        can_triage_jobs=False,
        can_edit_settings=False,
        can_create_users=False,
        can_create_groups=False,
        can_edit_roles=False,
        can_manage_infrastructure=False,
    )
    db.session.add(new_role)

    new_role = Role(
        name="Compliance Officer",
        can_triage_jobs=True,
        can_edit_settings=False,
        can_create_users=False,
        can_create_groups=False,
        can_edit_roles=False,
        can_manage_infrastructure=False,
    )
    db.session.add(new_role)

    new_role = Role(
        name="Administrator",
        can_triage_jobs=True,
        can_edit_settings=True,
        can_create_users=True,
        can_create_groups=True,
        can_edit_roles=False,
        can_manage_infrastructure=False,
    )
    db.session.add(new_role)

    new_role = Role(
        name="Owner",
        can_triage_jobs=True,
        can_edit_settings=True,
        can_create_users=True,
        can_create_groups=True,
        can_edit_roles=True,
        can_manage_infrastructure=True,
    )
    db.session.add(new_role)


def create_app(node_id: str, debug=False, n_replica=None, test_config=None) -> Flask:
    """Create flask application.

    Args:
         node_id: ID used to identify this node.
         debug: debug mode flag.
         n_replica: Number of model replicas used for fault tolerance purposes.
         test_config: database test settings.
    Returns:
         app : Flask App instance.
    """
    app = Flask(__name__)
    app.debug = debug

    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", None)

    if app.config["SECRET_KEY"] is None:
        app.config["SECRET_KEY"] = DEFAULT_SECRET_KEY
        logging.warning(
            "Using default secrect key, this is not safe and should be used only for testing and development. To define a secrete key please define the environment variable SECRET_KEY."
        )

    app.config["N_REPLICA"] = n_replica
    sockets = Sockets(app)

    # Register app blueprints
    from .main import (
        auth,
        data_centric_routes,
        hook,
        local_worker,
        main_routes,
        model_centric_routes,
        ws,
    )

    # set_node_id(id)
    local_worker.id = node_id
    hook.local_worker._known_workers[node_id] = local_worker
    local_worker.add_worker(hook.local_worker)

    # Register app blueprints
    app.register_blueprint(main_routes, url_prefix=r"/")
    app.register_blueprint(model_centric_routes, url_prefix=r"/model-centric")
    app.register_blueprint(data_centric_routes, url_prefix=r"/data-centric")

    sockets.register_blueprint(ws, url_prefix=r"/")

    # Set SQLAlchemy configs
    set_database_config(app, test_config=test_config)
    s = app.app_context().push()

    if database_exists(db.engine.url):
        db.create_all()
    else:
        db.create_all()
        seed_db()

    db.session.commit()

    # Set Authentication configs
    app = auth.set_auth_configs(app)

    CORS(app)

    # Threads
    executor.init_app(app)
    app.config["EXECUTOR_PROPAGATE_EXCEPTIONS"] = True
    app.config["EXECUTOR_TYPE"] = "thread"

    return app
