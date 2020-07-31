"""The purpose of this application is to allow us to dev and test PySyft
functionality on an actual local network. This is NOT meant to be run in
production (that's the *actual* grid's job)."""

from flask import Flask

app = Flask(__name__)

from syft.core.nodes.device.device import Device
import pickle

device = Device(name="cpu1")


def run():
    app.run()
