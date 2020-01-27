from flask import Flask
from flask_sockets import Sockets


def create_app(node_id, debug=False, database_url=None):
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

    # Enable persistent mode
    # Overwrite syft.object_storage methods to work in a persistent way
    # Persist models / tensors
    if database_url:
        app.config["REDISCLOUD_URL"] = database_url
        from .main.persistence import database, object_storage

        db_instance = database.set_db_instance(database_url)
        object_storage.set_persistent_mode(db_instance)

    from .main import html, ws, hook, local_worker, auth

    # Global socket handler
    sockets = Sockets(app)

    # set_node_id(id)
    local_worker.id = node_id
    hook.local_worker._known_workers[node_id] = local_worker

    # Register app blueprints
    app.register_blueprint(html, url_prefix=r"/")
    sockets.register_blueprint(ws, url_prefix=r"/")

    # Set Authentication configs
    app = auth.set_auth_configs(app)

    return app
