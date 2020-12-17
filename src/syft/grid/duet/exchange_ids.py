# stdlib
import json
from pathlib import Path
import tempfile
import time
from typing import Any as TypeAny

# syft relative
from .bcolors import bcolors


class DuetCredentialExchanger:
    def server_exchange(self, credential: TypeAny) -> TypeAny:
        raise NotImplementedError

    def client_exchange(self, credential: TypeAny) -> TypeAny:
        raise NotImplementedError


class OpenGridTokenManualInputExchanger(DuetCredentialExchanger):
    def server_exchange(self, credential: str) -> str:
        # send Server ID
        print()
        print(
            "♫♫♫ > "
            + bcolors.HEADER
            + "STEP 1:"
            + bcolors.ENDC
            + " Send the following Duet Server ID to your Duet Partner!"
        )
        #         print(f"♫♫♫ > Duet Node ID:{domain.id.value}")
        print("♫♫♫ > Duet Server ID: " + bcolors.BOLD + credential + bcolors.ENDC)
        # get Client ID
        print(
            "\n♫♫♫ > "
            + bcolors.HEADER
            + "STEP 2:"
            + bcolors.ENDC
            + " Have your duet partner send Client ID to you and enter it below!"
        )
        while True:
            target_id = input("♫♫♫ > Duet Partner's Client ID: ")  # nosec
            if len(target_id) == 32:
                break
            else:
                print("    > Error: Invalid Client ID. Please try again.")
        print()
        return target_id

    def client_exchange(self, credential: str) -> str:
        # get server ID
        print(
            "\n♫♫♫ > "
            + bcolors.HEADER
            + "STEP 1:"
            + bcolors.ENDC
            + " Have your duet partner send an Server ID to you and enter it below!"
        )
        while True:
            target_id = input("♫♫♫ > Duet Partner's Server ID: ")  # nosec
            if len(target_id) == 32:
                break
            else:
                print("    > Error: Invalid Server ID. Please try again.")

        # send client ID

        print()
        print(
            "♫♫♫ > "
            + bcolors.HEADER
            + "STEP 2:"
            + bcolors.ENDC
            + " Send the following Duet Client ID to your duet partner!"
        )
        print("♫♫♫ > Duet Client ID: " + bcolors.BOLD + credential + bcolors.ENDC)
        print()
        print("♫♫♫ > ...waiting for partner to connect...")

        return target_id


# for local debugging
def get_loopback_path() -> str:
    loopback_file = "duet_loopback.json"
    return str(Path(tempfile.gettempdir()) / loopback_file)


class OpenGridTokenFileExchanger(DuetCredentialExchanger):
    def __init__(self, file_path: str = get_loopback_path()):
        self.file_path = file_path

    def server_exchange(self, credential: str) -> str:
        print()
        print(
            "♫♫♫ > "
            + "Running loopback mode. Use sy.join_duet(loopback=True) on the other side."
        )
        print()

        # send Server ID
        loopback_config = {}
        loopback_config["server_id"] = credential
        with open(self.file_path, "w") as f:
            f.write(json.dumps(loopback_config))

        # get Client ID
        target_id = ""
        while target_id == "":
            try:
                with open(self.file_path, "r") as f:
                    loopback_config = json.loads(f.read())
                    if "client_id" in loopback_config:
                        target_id = str(loopback_config["client_id"])
                    else:
                        time.sleep(0.5)
            except Exception as e:
                print(e)
                break
        return target_id

    def client_exchange(self, credential: str) -> str:
        loopback_config = {}
        target_id = ""
        while target_id == "":
            try:
                with open(self.file_path, "r") as f:
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

        loopback_config["client_id"] = credential

        with open(self.file_path, "w") as f:
            f.write(json.dumps(loopback_config))

        return target_id


manual_exchanger = OpenGridTokenManualInputExchanger()
file_exchanger = OpenGridTokenFileExchanger()
