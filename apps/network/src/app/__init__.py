"""This package set up the app and server."""

import json
import os

from flask import Blueprint, Flask, Response
from flask_lambda import FlaskLambda
from flask_migrate import Migrate
from flask_sockets import Sockets
from flask_sqlalchemy import SQLAlchemy
from gevent import pywsgi
from geventwebsocket.handler import WebSocketHandler
from sqlalchemy_utils.functions import database_exists

# Set routes/events
ws = Blueprint(r"ws", __name__)
http = Blueprint(r"http", __name__)

db = SQLAlchemy()

from . import utils  # isort:skip
from . import routes, events  # isort:skip
from .database import Role  # isort:skip

DEFAULT_SECRET_KEY = "justasecretkeythatishouldputhere"
__version__ = "0.1.0"


def set_database_config(app, db_config=None, verbose=False):
    """Set configs to use SQL Alchemy library.

    Args:
        app: Flask application.
        db_config : Dictionary containing SQLAlchemy configs for test purposes.
        verbose : Level of flask application verbosity.

    Returns:
        app: Flask application.

    Raises:
        RuntimeError : If DATABASE_URL or db_config didn't initialize, RuntimeError exception will be raised.
    """
    db_url = os.environ.get("DATABASE_URL")
    migrate = Migrate(app, db)
    if db_config is None:
        if db_url:
            app.config.from_mapping(
                SQLALCHEMY_DATABASE_URI=db_url, SQLALCHEMY_TRACK_MODIFICATIONS=False
            )
        else:
            raise RuntimeError(
                "Invalid database address: Set DATABASE_URL environment var or add db_config parameter at create_app method."
            )
    else:
        app.config["SQLALCHEMY_DATABASE_URI"] = db_config["SQLALCHEMY_DATABASE_URI"]
        app.config["TESTING"] = (
            db_config["TESTING"] if db_config.get("TESTING") else True
        )
        app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = (
            db_config["SQLALCHEMY_TRACK_MODIFICATIONS"]
            if db_config.get("SQLALCHEMY_TRACK_MODIFICATIONS")
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


def create_app(debug=False, secret_key=DEFAULT_SECRET_KEY, db_config=None) -> Flask:
    """Create Flask app.

    Args:
        debug (bool): Enable debug mode
        secret_key (str): Secret key application
        db_config (Union[None, dict]): Database configuration

    Returns:
        app (Flask): flask application
    """
    app = Flask(__name__)
    app.debug = debug
    app.config["SECRET_KEY"] = secret_key

    # Global socket handler
    sockets = Sockets(app)

    app.register_blueprint(http, url_prefix=r"/")
    sockets.register_blueprint(ws, url_prefix=r"/")

    # Set SQLAlchemy configs
    global db
    set_database_config(app, db_config=db_config)
    s = app.app_context().push()

    if database_exists(db.engine.url):
        db.create_all()
    else:
        db.create_all()
        seed_db()

    db.session.commit()

    return app


def raise_grid(host: str, port: int, **kwargs):
    """Raise webserver application.

    Args:
        host (str): Hostname
        port (int): Port number
        **kwargs: Arbitrary keywords argument

    Returns:
        (tuple) tuple containing
            app (Flask): flask application
            server (pywsgi.WSGIServer): webserver
    """
    app = create_app(**kwargs)
    server = pywsgi.WSGIServer((host, port), app, handler_class=WebSocketHandler)
    server.serve_forever()
    return app, server


def create_lambda_app() -> FlaskLambda:
    """Create Flask Lambda app for deploying on AWS.

    Returns:
        app (Flask): flask application
    """

    database_name = os.environ.get("DB_NAME")
    cluster_arn = os.environ.get("DB_CLUSTER_ARN")
    secret_arn = os.environ.get("DB_SECRET_ARN")
    secret_key = os.environ.get("SECRET_KEY", DEFAULT_SECRET_KEY)

    app = FlaskLambda(__name__)
    app.config.from_mapping(
        DEBUG=False,
        SECRET_KEY=secret_key,
        SQLALCHEMY_DATABASE_URI=f"mysql+auroradataapi://:@/{database_name}",
        SQLALCHEMY_ENGINE_OPTIONS={
            "connect_args": dict(aurora_cluster_arn=cluster_arn, secret_arn=secret_arn)
        },
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
    )

    # Add a test route
    @http.route("/test-deployment/")
    def test():
        return Response(
            json.dumps({"message": "Serverless deployment successful"}),
            status=200,
            mimetype="application/json",
        )

    app.register_blueprint(http, url_prefix=r"/")

    # Setup database
    global db
    db.init_app(app)

    s = app.app_context().push()  # Push the app into context

    try:
        db.Model.metadata.create_all(
            db.engine, checkfirst=False
        )  # Create database tables
        seed_db()  # Seed the database
    except Exception as e:
        print("Error", e)

    db.session.commit()

    return app
