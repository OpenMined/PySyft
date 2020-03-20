import subprocess
import sys
import os

if os.name == "nt":
    python = "python"
else:
    python = "python" + sys.version[0:3]

FILE_PATH = os.path.abspath(__file__)
FILE_PATH = "/".join(FILE_PATH.split("/")[:-5]) + "/run_websocket_server.py"

call_alice = [python, FILE_PATH, "--port", "8777", "--id", "alice"]

call_bob = [python, FILE_PATH, "--port", "8778", "--id", "bob"]

call_charlie = [python, FILE_PATH, "--port", "8779", "--id", "charlie"]


print("Starting server for Alice")
subprocess.Popen(call_alice)

print("Starting server for Bob")
subprocess.Popen(call_bob)

print("Starting server for Charlie")
subprocess.Popen(call_charlie)
