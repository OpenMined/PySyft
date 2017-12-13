import zmq
import uuid

identity = str(uuid.uuid4())
context = zmq.Context()

socket = context.socket(zmq.DEALER)
socket.setsockopt_string(zmq.IDENTITY, identity)
socket.connect("tcp://localhost:5555")
verbose = False

def log(message):
    if (verbose):
        print(message)

print("reloaded")
