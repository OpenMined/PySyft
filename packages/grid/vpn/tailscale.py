# stdlib
import os
from typing import Dict

# third party
from flask import Flask
from flask_executor import Executor
from flask_executor.futures import Future
from flask_shell2http import Shell2HTTP

# Flask application instance
app = Flask(__name__)

executor = Executor(app)
shell2http = Shell2HTTP(app=app, executor=executor, base_url_prefix="/commands/")

hostname = os.environ.get("HOSTNAME", "node")  # default to node

key = os.environ.get("STACK_API_KEY",None) # Get key from environment
if key is None:
    sys.exit(1)

def check_key_callback(context: Dict, future: Future):
    if context["key"] != key:
        sys.exit(1)
    print(context, future.result())

def up_callback(context: Dict, future: Future) -> None:
    # optional user-defined callback function
    print(context, future.result())


shell2http.register_command(
    endpoint="up",
    command_name="tailscale up",
    callback_fn=up_callback,
    decorators=[],
)


def status_callback(context: Dict, future: Future) -> None:
    # optional user-defined callback function
    print(context, future.result())


shell2http.register_command(
    endpoint="status",
    command_name="tailscale status",
    callback_fn=status_callback,
    decorators=[],
)
shell2http.register_command(
    endpoint="check_key",
    command_name=f"echo {key}",
    callback_fn=check_key_callback,
    decorators=[],
)
