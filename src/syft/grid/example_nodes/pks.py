"""The purpose of this application is to allow us to dev and test PySyft
functionality on an actual local network. This is NOT meant to be run in
production (that's the *actual* grid's job)."""

# third party
from flask import Flask

# syft absolute
from syft.core.node.network.network import Network

app = Flask(__name__)
network = Network(name="ucsf-net")


@app.route("/")
def hello_world() -> str:
    return "A network:" + str(network.name)


def run() -> None:
    app.run()


run()
