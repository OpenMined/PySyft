"""
Note:

This file should be used only for development purposes.
Use the Flask built-in web server isn't suitable for production.
For production, we need to put it behind real web server able to communicate
with Flask through a WSGI protocol.
A common choice for that is Gunicorn.
"""

from app import create_app
from gevent import pywsgi
from geventwebsocket.handler import WebSocketHandler


if __name__ == "__main__":
    app = create_app()

    server = pywsgi.WSGIServer(("0.0.0.0", 5000), app, handler_class=WebSocketHandler)

    server.serve_forever()
