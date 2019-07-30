from flask import Flask
from flask_socketio import SocketIO
from flask_cors import CORS

socketio = SocketIO(async_mode="eventlet")


def create_app(debug=False):
    """Create flask socket-io application."""
    app = Flask(__name__)
    app.debug = debug
    app.config["SECRET_KEY"] = "justasecretkeythatishouldputhere"

    from .main import main as main_blueprint

    app.register_blueprint(main_blueprint)
    CORS(app)
    socketio.init_app(app)
    return app
