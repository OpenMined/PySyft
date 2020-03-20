import subprocess
import sys
import os
from pathlib import Path

if os.name == "nt":
    python = "python"
else:
    python = "python" + sys.version[0:3]

FILE_PATH = (
    Path(__file__)
    .resolve()
    .parent.parent.parent.parent.parent.parent.joinpath("run_websocket_server.py")
)

call_alice = [python, FILE_PATH, "--port", "8777", "--id", "alice", "--notebook", "steal_data"]

call_bob = [python, FILE_PATH, "--port", "8778", "--id", "bob", "--notebook", "steal_data"]

call_charlie = [python, FILE_PATH, "--port", "8779", "--id", "charlie", "--notebook", "steal_data"]


print("Starting server for Alice")
subprocess.Popen(call_alice)

print("Starting server for Bob")
subprocess.Popen(call_bob)

print("Starting server for Charlie")
subprocess.Popen(call_charlie)
