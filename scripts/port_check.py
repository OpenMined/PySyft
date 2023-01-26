"""Script to pause execution until uvicorn start for oblv tox tests"""
# stdlib
from multiprocessing import Process
from multiprocessing import set_start_method
import platform
import sys
import time

# third party
import requests
from requests.exceptions import ConnectionError
import uvicorn

port = int(sys.argv[1])


def start_uvicorn():
    uvicorn.run(app="app:app", port=port, host="0.0.0.0")


def check_uvicorn():
    TIMEOUT = 300  # 5 minutes
    ctr = 0
    while ctr <= TIMEOUT:
        try:
            res = requests.get(f"http://0.0.0.0:{port}/")
            if res.status_code == 200:
                break
        except ConnectionError:
            time.sleep(1)
            ctr += 1
        if ctr % 5 == 0:
            print("Waiting for uvicorn process to start...")

    if ctr > TIMEOUT:
        print("Uvicorn process check timed out... ")
    else:
        print("Uvicorn process started ")


def os_name() -> str:
    os_name = platform.system()
    if os_name.lower() == "darwin":
        return "macOS"
    else:
        return os_name


if os_name() == "macOS":
    # set start method to fork in case of MacOS
    set_start_method("fork", force=True)

process1 = Process(target=start_uvicorn)
process2 = Process(target=check_uvicorn)
process1.daemon = True
process1.start()
process2.start()
process2.join()
process1.join()
