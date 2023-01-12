# stdlib
import os
from pathlib import Path

# third party
import capnp
from flask import Flask
from flask import request
from flask import send_from_directory


def get_capnp_schema(schema_file: str) -> type:
    here = os.path.dirname(__file__)
    root_dir = Path(here) / "."
    capnp_path = os.path.abspath(root_dir / schema_file)
    return capnp.load(str(capnp_path))


app = Flask(__name__)


def make_foo(bar: str) -> bytes:
    foo = get_capnp_schema("./address.capnp").Foo
    msg = foo.new_message()
    msg.bar = bar
    return msg.to_bytes()


def bytes_to_foo(blob: bytes):
    MAX_TRAVERSAL_LIMIT = 2**64 - 1
    foo = get_capnp_schema("./address.capnp").Foo
    with foo.from_bytes(  # type: ignore
        blob, traversal_limit_in_words=MAX_TRAVERSAL_LIMIT
    ) as msg:
        return msg


@app.route("/rcv")
def message():
    print("RECIEVE")
    x = make_foo("bar")
    print(type(x))
    headers = {"content-type": "application/octect-stream"}

    return x, 200, headers


@app.route("/send", methods=["POST"])
def rcv_message():
    print("SEND")
    print(request.headers)
    body = request.get_data()
    print("data", body)
    print("type", type(body))

    foo = bytes_to_foo(body)
    print("foo", foo, foo.bar)

    return "Echo"


@app.route("/")
def index_js():
    return send_from_directory("./", "index.html")


@app.route("/client.js")
def client_js():
    return send_from_directory("./", "client.js")


@app.route("/address.capnp.js")
def address_js():
    return send_from_directory("./", "address.capnp.js")


@app.route("/address.capnp.d.ts")
def address_ts():
    return send_from_directory("./", "address.capnp.d.ts")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
