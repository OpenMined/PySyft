# stdlib
import os
from typing import Dict

# third party
from flask import Flask
from flask_executor import Executor
from flask_executor.futures import Future
from flask_shell2http import Shell2HTTP


# creating a more rigorous class from the shell2http
def check_key():
    key = os.environ.get("STACK_API_KEY",None) # Get key from environment
    if key is None:
        sys.exit(1)


# Flask application instance
app = Flask(__name__)

executor = Executor(app)
shell2http = Shell2HTTP(app=app, executor=executor, base_url_prefix="/commands/")

network_name = os.environ.get("NETWORK_NAME", "omnet")  # default to omnet


key = os.environ.get("STACK_API_KEY",None) # Get key from environment
if key is None:
    sys.exit(1)

def check_key_callback(context: Dict, future: Future):
    if context["key"] != key:
        sys.exit(1)
    print(context, future.result())



def generate_key_callback(context: Dict, future: Future) -> None:
    # optional user-defined callback function
  #  check_key()
    print(context, future.result())


shell2http.register_command(
    endpoint="generate_key",
    command_name=f"headscale -n {network_name} preauthkeys create -o json",
    callback_fn=generate_key_callback,
    decorators=[],
)


def list_nodes_callback(context: Dict, future: Future) -> None:
    # optional user-defined callback function
    print(context, future.result())


shell2http.register_command(
    endpoint="list_nodes",
    command_name=f"headscale -n {network_name} nodes list -o json",
    callback_fn=list_nodes_callback,
    decorators=[],
)


shell2http.register_command(
    endpoint="check_key",
    command_name=f"echo {key}",
    callback_fn=check_key_callback,
    decorators=[],
)