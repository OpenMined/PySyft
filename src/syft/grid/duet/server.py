from werkzeug.serving import make_server
import threading

class ServerThread(threading.Thread):

    def __init__(self, app, host='127.0.0.1', port=5000):
        threading.Thread.__init__(self)
        self.srv = make_server(host, port, app)
        self.ctx = app.app_context()
        self.ctx.push()

    def run(self):
        self.srv.serve_forever()

    def shutdown(self):
        self.srv.shutdown()