import subprocess
import sys
from pathlib import Path

python = Path(sys.executable).name

FILE_PATH = (
    Path(__file__)
    .resolve()
    .parents[1]
    .joinpath("websockets_mnist")
    .joinpath("run_websocket_server.py")
)

subprocess.Popen([python, FILE_PATH])
