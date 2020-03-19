import subprocess
import sys
import os

if os.name == "nt":
    python = "python"
else:
    python = "python" + sys.version[0:3]

call_alice = [python, "../../../../run_websocket_server.py", "--port", "8777", "--id", "alice", "--type","steal_data"]

call_bob = [python, "../../../../run_websocket_server.py", "--port", "8778", "--id", "bob", "--type","steal_data"]

call_charlie = [python, "../../../../run_websocket_server.py", "--port", "8779", "--id", "charlie", "--type","steal_data"]


print("Starting server for Alice")
subprocess.Popen(call_alice)

print("Starting server for Bob")
subprocess.Popen(call_bob)

print("Starting server for Charlie")
subprocess.Popen(call_charlie)
