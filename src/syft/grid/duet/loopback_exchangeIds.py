# stdlib
import json
from pathlib import Path
import tempfile
import time


# for local debugging
def get_loopback_path() -> str:
    loopback_file = "duet_loopback.json"
    return str(Path(tempfile.gettempdir()) / loopback_file)


def server_send_then_get(id_to_send: str) -> str:
    print()
    print(
        "♫♫♫ > "
        + "Running loopback mode. Use sy.join_duet(loopback=True) on the other side."
    )
    print()

    # send Server ID
    loopback_config = {}
    loopback_config["server_id"] = id_to_send
    with open(get_loopback_path(), "w") as f:
        f.write(json.dumps(loopback_config))

    # get Client ID
    target_id = ""
    while target_id == "":
        try:
            with open(get_loopback_path(), "r") as f:
                loopback_config = json.loads(f.read())
                if "client_id" in loopback_config:
                    target_id = str(loopback_config["client_id"])
                else:
                    time.sleep(0.5)
        except Exception as e:
            print(e)
            break
    return target_id


def client_get_then_send(id_to_send: str) -> str:
    loopback_config = {}
    target_id = ""
    while target_id == "":
        try:
            with open(get_loopback_path(), "r") as f:
                loopback_config = json.loads(f.read())
                # only continue once the server has overwritten the file
                # with only its new server_id
                if (
                    "server_id" in loopback_config
                    and "client_id" not in loopback_config
                ):
                    target_id = str(loopback_config["server_id"])
                else:
                    time.sleep(0.5)
        except Exception as e:
            print(e)
            break

    loopback_config["client_id"] = id_to_send

    with open(get_loopback_path(), "w") as f:
        f.write(json.dumps(loopback_config))

    return target_id
