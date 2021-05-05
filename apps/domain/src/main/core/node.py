# third party
from flask_sockets import Sockets
from main import ws
from nacl.encoding import HexEncoder
from nacl.signing import SigningKey

# grid relative
from ..routes import association_requests_blueprint
from ..routes import dcfl_blueprint
from ..routes import groups_blueprint
from ..routes import mcfl_blueprint
from ..routes import roles_blueprint
from ..routes import root_blueprint
from ..routes import search_blueprint
from ..routes import setup_blueprint
from ..routes import users_blueprint
from ..utils.executor import executor
from .nodes.domain import GridDomain
from .nodes.network import GridNetwork
from .nodes.worker import GridWorker
from .sleepy_until_configured import SleepyUntilConfigured

node = None


def get_node():
    global node
    return node


def create_worker_app(app, args):
    # Register HTTP blueprints
    # Here you should add all the blueprints related to HTTP routes.
    app.register_blueprint(root_blueprint, url_prefix=r"/")

    # Register WebSocket blueprints
    # Here you should add all the blueprints related to WebSocket routes.

    global node
    node = GridWorker(name=args.name, domain_url=args.domain_address)

    app.config["EXECUTOR_PROPAGATE_EXCEPTIONS"] = True
    app.config["EXECUTOR_TYPE"] = "thread"
    executor.init_app(app)

    return app


def create_network_app(app, args, testing=False):
    test_config = None
    if args.start_local_db:
        test_config = {"SQLALCHEMY_DATABASE_URI": "sqlite:///nodedatabase.db"}

    app.register_blueprint(roles_blueprint, url_prefix=r"/roles")
    app.register_blueprint(users_blueprint, url_prefix=r"/users")
    app.register_blueprint(setup_blueprint, url_prefix=r"/setup")
    app.register_blueprint(root_blueprint, url_prefix=r"/")
    app.register_blueprint(search_blueprint, url_prefix=r"/search")
    app.register_blueprint(
        association_requests_blueprint, url_prefix=r"/association-requests/"
    )

    # Register WebSocket blueprints
    # Here you should add all the blueprints related to WebSocket routes.

    # grid relative
    from .database import Role
    from .database import User
    from .database import db
    from .database import seed_db
    from .database import set_database_config

    global node
    node = GridNetwork(name=args.name)

    # Set SQLAlchemy configs
    set_database_config(app, test_config=test_config)
    s = app.app_context().push()

    db.create_all()

    if not testing:
        if len(db.session.query(Role).all()) == 0:
            seed_db()

        role = db.session.query(Role.id).filter_by(name="Owner").first()
        user = User.query.filter_by(role=role.id).first()
        if user:
            signing_key = SigningKey(
                user.private_key.encode("utf-8"), encoder=HexEncoder
            )
            node.signing_key = signing_key
            node.verify_key = node.signing_key.verify_key
            node.root_verify_key = node.verify_key
    db.session.commit()

    app.config["EXECUTOR_PROPAGATE_EXCEPTIONS"] = True
    app.config["EXECUTOR_TYPE"] = "thread"
    executor.init_app(app)

    return app


def create_domain_app(app, args, testing=False):
    test_config = None

    if args.start_local_db:
        test_config = {"SQLALCHEMY_DATABASE_URI": "sqlite:///nodedatabase.db"}

    # Bind websocket in Flask app instance
    sockets = Sockets(app)

    # Register HTTP blueprints
    # Here you should add all the blueprints related to HTTP routes.
    app.register_blueprint(roles_blueprint, url_prefix=r"/roles")
    app.register_blueprint(users_blueprint, url_prefix=r"/users")
    app.register_blueprint(setup_blueprint, url_prefix=r"/setup")
    app.register_blueprint(groups_blueprint, url_prefix=r"/groups")
    app.register_blueprint(dcfl_blueprint, url_prefix=r"/data-centric/")
    app.register_blueprint(mcfl_blueprint, url_prefix=r"/model-centric/")
    app.register_blueprint(root_blueprint, url_prefix=r"/")
    app.register_blueprint(
        association_requests_blueprint, url_prefix=r"/association-requests/"
    )

    # Register WebSocket blueprints
    # Here you should add all the blueprints related to WebSocket routes.
    sockets.register_blueprint(ws, url_prefix=r"/")

    # grid relative
    from .database import Role
    from .database import SetupConfig
    from .database import User
    from .database import db
    from .database import seed_db
    from .database import set_database_config

    global node
    node = GridDomain(name=args.name)

    # Set SQLAlchemy configs
    set_database_config(app, test_config=test_config)
    app.app_context().push()
    db.create_all()

    if not testing:
        if len(db.session.query(Role).all()) == 0:
            seed_db()

        if len(db.session.query(SetupConfig).all()) != 0:
            node.name = db.session.query(SetupConfig).first().domain_name

        role = db.session.query(Role.id).filter_by(name="Owner").first()
        user = User.query.filter_by(role=role.id).first()
        if user:
            signing_key = SigningKey(
                user.private_key.encode("utf-8"), encoder=HexEncoder
            )
            node.signing_key = signing_key
            node.verify_key = node.signing_key.verify_key
            node.root_verify_key = node.verify_key

        # Register global middlewares
        # Always after context is pushed
        app.wsgi_app = SleepyUntilConfigured(app, app.wsgi_app)
    db.session.commit()

    app.config["EXECUTOR_PROPAGATE_EXCEPTIONS"] = True
    app.config["EXECUTOR_TYPE"] = "thread"
    executor.init_app(app)

    return app
