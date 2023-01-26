"""Script to pause execution until uvicorn start for oblv tox tests"""
# stdlib
import socket as sock
import sys
import time

port = int(sys.argv[1])
TIMEOUT = 300  # 5 minutes
ctr = 0

while ctr <= TIMEOUT:
    socket = sock.socket(sock.AF_INET, sock.SOCK_STREAM)
    res = socket.connect_ex(("0.0.0.0", port))
    if res == 0:
        break
    else:
        time.sleep(1)
        ctr += 1
    if ctr % 5 == 0:
        print("Waiting for uvicorn process to start...")
    socket.close()

if ctr > TIMEOUT:
    print("Uvicorn process check timed out... ")
else:
    print("Uvicorn process started ")
