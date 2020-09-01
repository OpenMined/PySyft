"""The purpose of this application is to allow us to dev and test PySyft
functionality on an actual local network. This is NOT meant to be run in
production (that's the *actual* grid's job)."""

# stdlib
import pickle

# third party
from flask import Flask

# syft relative
from ..core.node.network.network import Network

app = Flask(__name__)

network = Network(name="ucsf-net")


@app.route("/")
def get_client() -> str:
    client = network.get_client()
    return pickle.dumps(client).hex()


def run() -> None:
    app.run()
