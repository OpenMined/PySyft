from flask import Flask
from flask import request
import torch as th
from torch import nn
import syft as sy

sy.create_sandbox(globals())

app = Flask(__name__)

# Iniitalize A Toy Model
model = th.zeros([2, 1])
ptr = None


@app.route("/get_model", methods=["GET"])
def get_model():
    global model
    global ptr
    ptr = model.create_pointer()
    return model.ser()


@app.route("/send_data", methods=["POST"])
def send_data():
    ptr = sy.serde.deserialize(request.data)

    return model.ser()
