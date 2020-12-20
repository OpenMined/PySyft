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
from nacl.signing import SigningKey
from nacl.encoding import HexEncoder
from syft.core.node.domain.domain import Domain


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

DEFAULT_SECRET_KEY = "justasecretkeythatishouldputhere"

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


def create_app(
    test_config: Optional[Dict] = None, secret_key=DEFAULT_SECRET_KEY, debug=False
) -> Flask:
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
    app.register_blueprint(roles_blueprint, url_prefix=r"/roles")
    app.register_blueprint(users_blueprint, url_prefix=r"/users")
    app.register_blueprint(setup_blueprint, url_prefix=r"/setup/")
    app.register_blueprint(groups_blueprint, url_prefix=r"/groups")
    app.register_blueprint(dcfl_blueprint, url_prefix=r"/dcfl/")
    app.register_blueprint(root_blueprint, url_prefix=r"/")
    app.register_blueprint(
        association_requests_blueprint, url_prefix=r"/association-requests/"
    )

    # Register WebSocket blueprints
    # Here you should add all the blueprints related to WebSocket routes.
    # sockets.register_blueprint()

    app.debug = debug
    app.config["SECRET_KEY"] = secret_key

    from main.core.database import db, set_database_config, seed_db, User, Role
    from main.core.node import node
    from main.core.task_handler import executor

    # Set SQLAlchemy configs
    set_database_config(app, test_config=test_config)
    s = app.app_context().push()

    db.create_all()

    if not app.config["TESTING"]:
        if len(db.session.query(Role).all()) == 0:
            seed_db()

        role = db.session.query(Role.id).filter_by(name="Owner").first()
        user = User.query.filter_by(role=role.id).first()
        if user:
            global node
            signing_key = SigningKey(
                user.private_key.encode("utf-8"), encoder=HexEncoder
            )
            node.signing_key = signing_key
            node.verify_key = node.signing_key.verify_key
            node.root_verify_key = node.verify_key
    db.session.commit()

    # Threads
    executor.init_app(app)
    app.config["EXECUTOR_PROPAGATE_EXCEPTIONS"] = True
    app.config["EXECUTOR_TYPE"] = "thread"

    # Send app instance
    return app
