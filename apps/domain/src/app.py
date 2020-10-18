"""CODING GUIDELINES:

- Add docstrings following the pattern chosen by the community.
- Add comments explaining step by step how your method works and the purpose of it.
- If possible, add examples showing how to call them properly.
- Remember to add the parameters and return types.
- Add unit tests / integration tests for every feature that you develop in order to cover at least 80% of the code.
- Import order : python std libraries, extendend libs, internal source code.
"""

# Std Python imports
from typing import Optional
from typing import Dict
import logging
import os

# Extended Python imports
from flask import Flask
from flask_sockets import Sockets
from geventwebsocket.websocket import Header
from sqlalchemy_utils.functions import database_exists

# Internal imports
from main.utils.monkey_patch import mask_payload_fast
from main.routes import (
    roles_blueprint,
    users_blueprint,
    setup_blueprint,
    groups_blueprint,
    dcfl_blueprint,
    association_requests_blueprint,
    root_blueprint,
)
import config

# Masking/Unmasking is a process used to guarantee some level of security
# during the transportation of the messages across proxies (as described in WebSocket RFC).
# Since the masking process needs to iterate over the message payload,
# the larger this message is, the longer it takes to process it.
# The flask_sockets / gevent-websocket developed this process using only native language structures.
# Replacing these structures for NumPy structures should increase the performance.
Header.mask_payload = mask_payload_fast
Header.unmask_payload = mask_payload_fast

# Setup log
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s]: {} %(levelname)s %(message)s".format(os.getpid()),
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger()


def create_app(test_config: Optional[Dict] = None) -> Flask:
    """This method creates a new Flask App instance and attach it with some
    HTTP/Websocket bluetprints.

    PS: In order to keep modularity and reause, do not add any PyGrid logic here, this method should be as logic agnostic as possible.
    :return: returns a Flask app instance.
    :rtype: Flask
    """
    logger.info(f"Starting app in {config.APP_ENV} environment")

    # Create Flask app instance
    app = Flask(__name__)

    app.config.from_object("config")

    # Bind websocket in Flask app instance
    sockets = Sockets(app)

    # Register HTTP blueprints
    # Here you should add all the blueprints related to HTTP routes.
    app.register_blueprint(roles_blueprint, url_prefix=r"/roles/")
    app.register_blueprint(users_blueprint, url_prefix=r"/users/")
    app.register_blueprint(setup_blueprint, url_prefix=r"/setup/")
    app.register_blueprint(groups_blueprint, url_prefix=r"/groups/")
    app.register_blueprint(dcfl_blueprint, url_prefix=r"/dcfl/")
    app.register_blueprint(root_blueprint, url_prefix=r"/")
    app.register_blueprint(
        association_requests_blueprint, url_prefix=r"/association-requests/"
    )

    # Register WebSocket blueprints
    # Here you should add all the blueprints related to WebSocket routes.
    # sockets.register_blueprint()

    from main.core.database import db, set_database_config, seed_db

    # Set SQLAlchemy configs
    set_database_config(app, test_config=test_config)
    s = app.app_context().push()

    if database_exists(db.engine.url):
        db.create_all()
    else:
        db.create_all()
        seed_db()

    db.session.commit()

    # Send app instance
    return app
