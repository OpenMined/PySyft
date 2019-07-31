import os
from flask_migrate import Migrate
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

from lib.models import db
from lib.worker_router import worker_router_bp


def create_app(test_config=None, verbose=False):
    global db
    app = Flask(__name__)
    db_url = os.environ.get("DATABASE_URL")
    migrate = Migrate(app, db)
    if test_config is None:
        app.config.from_mapping(
            SQLALCHEMY_DATABASE_URI=db_url, SQLALCHEMY_TRACK_MODIFICATIONS=False
        )

    else:
        # load the test config if passed in
        app.config["SQLALCHEMY_DATABASE_URI"] = test_config["SQLALCHEMY_DATABASE_URI"]
        app.config["TESTING"] = True
        app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["VERBOSE"] = verbose
    db.init_app(app)
    app.register_blueprint(worker_router_bp)
    app.add_url_rule("/", endpoint="index")
    return app
