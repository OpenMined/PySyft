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
shell2http = Shell2HTTP(app=app, executor=executor, base_url_prefix="/commands/")

hostname = os.environ.get("HOSTNAME", "node")  # default to node


key = os.environ.get("STACK_API_KEY", None)  # Get key from environment
if key is None:
    print("No STACK_API_KEY found, exiting.")
    sys.exit(1)

def check_stack_api_key(challenge_key: str) -> bool:
    key = os.environ.get("STACK_API_KEY", None)  # Get key from environment
    if key is None:
        return False
    if challenge_key == key:
        return True
    return False

def basic_auth_check(f): 
     @functools.wraps(shell2http) 
     def inner_decorator(*args, **kwargs): 
         #stack_api_key = request.headers.get("X-STACK-API-KEY", None)
         if not check_stack_api_key(challenge_key=key):
             #os.abort(Response("You are not logged in.", 401))
             raise Exception("STACK_API_KEY doesn't match.")
         return shell2http
  
     return inner_decorator


def up_callback(context: Dict, future: Future) -> None:
    # optional user-defined callback function
    print(context, future.result())


shell2http.register_command(
    endpoint="up",
    command_name="tailscale up",
    callback_fn=up_callback,
    decorators=[basic_auth_check],
)


def status_callback(context: Dict, future: Future) -> None:
    # optional user-defined callback function
    print(context, future.result())


shell2http.register_command(
    endpoint="status",
    command_name="tailscale status",
    callback_fn=status_callback,
    decorators=[basic_auth_check],
)
