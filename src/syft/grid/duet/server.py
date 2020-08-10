from flask import Flask
from werkzeug.serving import make_server
import threading


class ServerThread(threading.Thread):
    def __init__(self, app: Flask, host: str = "127.0.0.1", port: int = 5000) -> None:
        threading.Thread.__init__(self)
        self.srv = make_server(host, port, app)
        self.ctx = app.app_context()
        self.ctx.push()

    def run(self) -> None:
        self.srv.serve_forever()

    def shutdown(self) -> None:
        self.srv.shutdown()
