import os
import redis
from flask import Flask, session, request
import json
import syft as sy
import torch as th
import binascii


hook = sy.TorchHook(th)

app = Flask(__name__)
app.secret_key = b'keepmesecret'

try:
    db=redis.from_url(os.environ['REDISCLOUD_URL'])
except:
    db=redis.from_url('http://127.0.0.1:6379')

@app.route('/')
def hello_world():
    name=db.get('name') or'World'
    db.set('del_ctr', 0)
    return 'Howdy %s!' % str(name)

@app.route("/identity/")
def is_this_an_opengrid_node():
    """This exists because in the automation scripts which deploy nodes,
    there's an edge case where the 'node already exists' but sometimes it
    can be an app that does something totally different. So we want to have
    some endpoint which just casually identifies this server as an OpenGrid
    server"""

    return "OpenGrid"

@app.route('/cmd/', methods=['POST'])
def cmd():
    try:
        worker = db.get('worker')

        if(worker is None):
            worker = sy.VirtualWorker(hook, "grid", auto_add=False)
            print("\t \nCREATING NEW WORKER!!")
        else:
            worker = sy.serde.deserialize(worker)
            print("\t \nFOUND OLD WORKER!! " + str(worker._objects.keys()))

        worker.verbose = True
        sy.torch.hook.local_worker.add_worker(worker)

        message = request.form['message']
        message = binascii.unhexlify(message[2:-1])

        response = worker._recv_msg(message)

        response = str(binascii.hexlify(response))

        print("\t NEW WORKER STATE:" + str(worker._objects.keys()) + "\n\n")
        db.set('worker', sy.serde.serialize(worker, force_full=True))

        return response
    except Exception as e:
        return str(e)

    #
# @app.route('/createworker/<name>')
# def create_worker(name):
#     try:
#         worker = sy.VirtualWorker(hook, name)
#         session['worker'] = sy.serde.serialize(worker, force_full=True)
#         return "Create worker with id " + str(name)
#     except Exception as e:
#         return str(e)
#
#
# @app.route('/getworker/')
# def get_worker():
#     worker_str = sy.serde.deserialize(session['worker']).id
#
#     return "the current session worker is:" + worker_str
#
# @app.route('/setname/<name>')
# def setname(name):
#
#     worker = sy.VirtualWorker(hook, name)
#
#     db.set('name',str(worker.id))
#     return 'Name updated to ' + str(worker.id) + ' or ' + str(name)

if __name__ == '__main__':
    app.run()