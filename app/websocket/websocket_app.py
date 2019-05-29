from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import syft as sy
import torch as th
import redis
import binascii
import os

hook = sy.TorchHook(th)

# Set up REDIS URL
try:
    db = redis.from_url(os.environ["REDISCLOUD_URL"])
except:
    db = redis.from_url("redis://localhost:6379")


# Initialize app/socket vars
app = Flask(__name__)
app.config["SECRET_KEY"] = "justasecretkeythatishouldputhere"
socketio = SocketIO(app, async_mode="eventlet")
CORS(app)


def _maybe_create_worker(worker_name: str = "worker", virtual_worker_id: str = "grid"):
    worker = db.get(worker_name)
    if worker is None:
        worker = sy.VirtualWorker(hook, virtual_worker_id, auto_add=False)
    else:
        worker = sy.serde.deserialize(worker)
    return worker


def _store_worker(worker, worker_name: str = "worker"):
    db.set(worker_name, sy.serde.serialize(worker, force_full_simplification=True))


def _request_message(worker, request):
    print("Message Type: ", type(request))
    message = request["message"]
    message = binascii.unhexlify(message[2:-1])
    response = worker._recv_msg(message)
    response = str(binascii.hexlify(response))
    return response


@app.route("/")
def hello_world():
    name = db.get("name") or "World"
    db.set("del_ctr", 0)
    return "Websocket Howdy %s!" % str(name)


@socketio.on("/identity")
def is_this_an_opengrid_node():
    """This exists because in the automation scripts which deploy nodes,
    there's an edge case where the 'node already exists' but sometimes it
    can be an app that does something totally different. So we want to have
    some endpoint which just casually identifies this server as an OpenGrid
    server."""
    socketio.emit("/identity", "Websocket OpenGrid")


@socketio.on("/cmd")
def cmd(message):
    try:
        worker = _maybe_create_worker("worker", "grid")

        worker.verbose = True
        sy.torch.hook.local_worker.add_worker(worker)

        response = _request_message(worker, message)

        _store_worker(worker, "worker")

        socketio.emit("/cmd", response)
    except Exception as e:
        socketio.emit("/cmd", str(e))


if __name__ == "__main__":
    socketio.run(app)
