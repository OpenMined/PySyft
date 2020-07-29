"""The purpose of this application is to allow us to dev and test PySyft
functionality on an actual local network. This is NOT meant to be run in
production (that's the *actual* grid's job)."""

from flask import Flask

app = Flask(__name__)

from syft.core.nodes.network.network import Network
import pickle

network = Network(name="ucsf-net")


@app.route("/")
def get_client():

    client = network.get_client()

    return pickle.dumps(client).hex()


def run():
    app.run()
