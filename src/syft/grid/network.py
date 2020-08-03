"""The purpose of this application is to allow us to dev and test PySyft
functionality on an actual local network. This is NOT meant to be run in
production (that's the *actual* grid's job)."""

import pickle

from flask import Flask

from syft.core.node.network.network import Network

app = Flask(__name__)


network = Network(name="ucsf-net")


@app.route("/")
def get_client():

    client = network.get_client()

    return pickle.dumps(client).hex()


def run():
    app.run()
