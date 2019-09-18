from flask import Flask
from flask_socketio import SocketIO

from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import os

# TODO: define a reasonable ping interval
# and ping timeout
PING_INTERVAL = 10000000
PING_TIMEOUT = 5000

from eventlet import monkey_patch

monkey_patch()  # Turn every blocking module used by this app as much as asynchronous as possible

socketio = SocketIO(
    async_mode="eventlet", ping_interval=PING_INTERVAL, ping_timeout=PING_TIMEOUT
)


def set_database_config(app, test_config=None, verbose=False):
    """ Set configs to use SQL Alchemy library.

        Args:
            app: Flask application.
            test_config : Dictionary containing SQLAlchemy configs for test purposes.
            verbose : Level of flask application verbosity.
        Returns:
            app: Flask application.
        Raises:
            RuntimeError : If DATABASE_URL or test_config didn't initialized, RuntimeError exception will be raised.
    """
    global db
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
    return app


def create_app(debug=False, test_config=None):
    """Create flask socket-io application."""
    app = Flask(__name__)
    app.debug = debug
    app.config["SECRET_KEY"] = "justasecretkeythatishouldputhere"

    from .main import main as main_blueprint
    from .main import db

    global db

    app.register_blueprint(main_blueprint)

    # Set SQLAlchemy configs
    app = set_database_config(app, test_config=test_config)
    s = app.app_context().push()
    db.create_all()
    socketio.init_app(app)

    return app
