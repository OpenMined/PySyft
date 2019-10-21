from flask import Flask
from flask_sockets import Sockets


def create_app(node_id, debug=False, test_config=None):
    """ Create / Configure flask socket application instance.
        
        Args:
            node_id (str) : ID of Grid Node.
            debug (bool) : debug flag.
            test_config (bool) : Mock database environment.
        Returns:
            app : Flask application instance.
    """
    app = Flask(__name__)
    app.debug = debug
    app.config["SECRET_KEY"] = "justasecretkeythatishouldputhere"

    from .main import html, ws, db, hook, local_worker, auth
    from .main.persistence.utils import set_database_config

    global db
    sockets = Sockets(app)

    # set_node_id(id)
    local_worker.id = node_id
    hook.local_worker._known_workers[node_id] = local_worker

    # Register app blueprints
    app.register_blueprint(html, url_prefix=r"/")
    sockets.register_blueprint(ws, url_prefix=r"/")

    # Set Authentication configs
    app = auth.set_auth_configs(app)

    # Set SQLAlchemy configs
    app = set_database_config(app, test_config=test_config)
    s = app.app_context().push()
    db.create_all()

    return app
