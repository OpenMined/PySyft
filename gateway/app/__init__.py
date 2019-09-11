import logging
import os

from flask import Flask
from flask_cors import CORS


# Default secret key used only for testing / development
DEFAULT_SECRET_KEY = "justasecretkeythatishouldputhere"


def create_app(debug=False, n_replica=None):
    """Create flask application."""
    app = Flask(__name__)
    app.debug = debug

    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", None)

    if app.config["SECRET_KEY"] is None:
        app.config["SECRET_KEY"] = DEFAULT_SECRET_KEY
        logging.warn(
            "Using default secrect key, this is not safe and should be used only for testing and development. To define a secrete key please define the environment variable SECRET_KEY."
        )

    app.config["N_REPLICA"] = n_replica

    from .main import main as main_blueprint

    app.register_blueprint(main_blueprint)
    CORS(app)
    return app
