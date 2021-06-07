"""CODING GUIDELINES:

- Add docstrings following the pattern chosen by the community.
- Add comments explaining step by step how your method works and the purpose of it.
- If possible, add examples showing how to call them properly.
- Remember to add the parameters and return types.
- Add unit tests / integration tests for every feature that you develop in order to cover at least 80% of the code.
- Import order : python std libraries, extendend libs, internal source code.
"""

# stdlib
import logging
import os

# third party
import config

# Extended Python imports
from flask import Flask
from flask_cors import CORS
from geventwebsocket.websocket import Header
from main.core.node import create_domain_app
from main.routes import association_requests_blueprint  # noqa: 401
from main.routes import dcfl_blueprint  # noqa: 401
from main.routes import groups_blueprint  # noqa: 401
from main.routes import roles_blueprint  # noqa: 401
from main.routes import root_blueprint  # noqa: 401
from main.routes import setup_blueprint  # noqa: 401
from main.routes import users_blueprint  # noqa: 401

# Internal imports
from main.utils.monkey_patch import mask_payload_fast

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
    args,
    secret_key=DEFAULT_SECRET_KEY,
    debug=False,
    testing=False,
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

    # Create Domain APP
    app = create_domain_app(app=app, args=args, testing=testing)
    CORS(app)

    app.debug = debug
    app.config["SECRET_KEY"] = secret_key

    # Send app instance
    return app
