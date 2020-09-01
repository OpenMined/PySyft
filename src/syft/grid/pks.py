"""The purpose of this application is to allow us to dev and test PySyft
functionality on an actual local network. This is NOT meant to be run in
production (that's the *actual* grid's job)."""

# third party
from flask import Flask

# syft relative
from ..core.node.network.network import Network

app = Flask(__name__)
network = Network(name="ucsf-net")


@app.route("/")
def hello_world() -> str:
    return "A network:" + str(network.name)


def run() -> None:
    app.run()


# import argparse
# parser = argparse.ArgumentParser(description='Add some integers.')
# parser.add_argument('integers', metavar='N', type=int, nargs='+',
#                     help='interger list')
# parser.add_argument('--sum', action='store_const',
#                     const=sum, default=max,
#                     help='sum the integers (default: find the max)')
# args = parser.parse_args()
# print(args.sum(args.integers))
