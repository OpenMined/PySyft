import subprocess
import sys
import os

if os.name == "nt":
    python = "python"
else:
    python = "python" + sys.version[0:3]

call_alice = [python, "run_websocket_server.py", "--port", "8777", "--id", "alice"]

call_bob = [python, "run_websocket_server.py", "--port", "8778", "--id", "bob"]

call_charlie = [python, "run_websocket_server.py", "--port", "8779", "--id", "charlie"]


print("Starting server for Alice")
subprocess.Popen(call_alice)

print("Starting server for Bob")
subprocess.Popen(call_bob)

print("Starting server for Charlie")
subprocess.Popen(call_charlie)
