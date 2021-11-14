# stdlib
import os
import sys
from typing import Dict

# third party
from flask import Flask
import functools
from flask import request
from flask.wrappers import Response
from flask_executor import Executor
from flask_executor.futures import Future
from flask_shell2http import Shell2HTTP

# Flask application instance
app = Flask(__name__)

executor = Executor(app)

def check_stack_api_key(challenge_key: str) -> bool:
    key = os.environ.get("STACK_API_KEY", None)  # Get key from environment
    if key is None:
        return False
    if challenge_key == key:
        return True
    return False




shell2http = Shell2HTTP(app=app, executor=executor, base_url_prefix="/commands/")

network_name = os.environ.get("NETWORK_NAME", "omnet")  # default to omnet

key = os.environ.get("STACK_API_KEY", None)  # Get key from environment
if key is None:
    print("No STACK_API_KEY found, exiting.")
    sys.exit(1)


def basic_auth_check(f): 
     @functools.wraps(shell2http) 
     def inner_decorator(*args, **kwargs): 
         #stack_api_key = request.headers.get("X-STACK-API-KEY", None)
         if not check_stack_api_key(challenge_key=key):
             #os.abort(Response("You are not logged in.", 401))
             raise Exception("STACK_API_KEY doesn't match.")
         return shell2http
  
     return inner_decorator


def generate_key_callback(context: Dict, future: Future) -> None:
    # optional user-defined callback function
    print(context, future.result())


shell2http.register_command(
    endpoint="generate_key",
    command_name=f"headscale -n {network_name} preauthkeys create -o json",
    callback_fn=generate_key_callback,
    decorators=[basic_auth_check],
)

def list_nodes_callback(context: Dict, future: Future) -> None:
    # optional user-defined callback function
    print(context, future.result())


shell2http.register_command(
    endpoint="list_nodes",
    command_name=f"headscale -n {network_name} nodes list -o json",
    callback_fn=list_nodes_callback,
    decorators=[basic_auth_check],
)
