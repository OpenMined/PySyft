"""The purpose of this application is to allow us to dev and test PySyft
functionality on an actual local network. This is NOT meant to be run in
production (that's the *actual* grid's job)."""


# third party
from flask import Flask

# syft absolute
from syft.core.node.device.device import Device

app = Flask(__name__)
device = Device(name="cpu1")


def run() -> None:
    app.run()


run()
