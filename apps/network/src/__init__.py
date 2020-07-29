"""This package set up the app and server."""

import os

from flask import Blueprint, Flask
from flask_migrate import Migrate
from flask_sockets import Sockets
from flask_sqlalchemy import SQLAlchemy
from gevent import pywsgi
from geventwebsocket.handler import WebSocketHandler

# Set routes/events
ws = Blueprint(r"ws", __name__)
http = Blueprint(r"http", __name__)
# Set db client instance
db = SQLAlchemy()

from . import utils  # isort:skip
from . import routes, events  # isort:skip

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
        RuntimeError : If DATABASE_URL or db_config didn't initialized, RuntimeError exception will be raised.
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
    db.create_all()
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
