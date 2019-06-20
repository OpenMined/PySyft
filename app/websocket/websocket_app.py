from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import grid as gr
import syft as sy
import torch as th
import redis
import binascii
import os
import sys

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


def _request_message(worker, request):
    message = request["message"]
    message = binascii.unhexlify(message[2:-1])
    response = worker._recv_msg(message)
    response = str(binascii.hexlify(response))
    return response


@app.route("/")
def hello_world():
    return "Websocket %s! Grid Node" % str(hook.local_worker.id)


@socketio.on("/identity")
def is_this_an_opengrid_node():
    """This exists because in the automation scripts which deploy nodes,
    there's an edge case where the 'node already exists' but sometimes it
    can be an app that does something totally different. So we want to have
    some endpoint which just casually identifies this server as an OpenGrid
    server."""
    socketio.emit("/identity", "Websocket OpenGrid")


@socketio.on("/set-grid-id")
def set_grid_name(msg):
    me = hook.local_worker
    me.id = msg["id"]
    me.is_client_worker = False


@socketio.on("/connect-node")
def connect_node(msg):
    try:
        new_worker = gr.WebsocketGridClient(hook, msg["uri"], id=msg["id"])
        new_worker.connect()
        socketio.emit("/connect-node", "Succefully connected!")
    except Exception as e:
        socketio.emit("/connect-node", str(e))


@socketio.on("/cmd")
def cmd(message):
    try:
        worker = hook.local_worker
        response = _request_message(worker, message)
        socketio.emit("/cmd", response)
    except Exception as e:
        socketio.emit("/cmd", str(e))


if __name__ == "__main__":
    socketio.run(app)
