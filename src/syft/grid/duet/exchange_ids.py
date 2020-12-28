# stdlib
import json
from pathlib import Path
import tempfile
import time
from typing import Any as TypeAny
from typing import Optional as TypeOptional
from typing import Tuple as TypeTuple

# syft relative
from .bcolors import bcolors


class DuetTokenExchanger:
    def __init__(self, *args: TypeTuple[TypeAny, ...], **kwargs: TypeAny) -> None:
        self.join = False

    def set_role(self, join: bool = False) -> "DuetTokenExchanger":
        self.join = join
        return self

    def set_responder_id(self, duet_token: TypeAny) -> "DuetTokenExchanger":
        self.responder_id = duet_token
        return self

    def run(self, duet_token: str) -> str:
        raise NotImplementedError


# class AriesCredentialExchanger(DuetCredentialExchanger):
#     def __init__(self, agent: TypeAny) -> None:
#         super().__init__()
#         self.agent = agent

#     def run(
#         self,
#         credential: str,
#     ) -> str:
#         self.requester_id = credential
#         self.agent.joiner(self.join)
#         self.agent.send(self.requester_id)
#         while True:
#             if self.agent.has_response():
#                 self.responder_id = self.agent.get_responder_id()

#         return self.responder_id


class OpenGridTokenManualInputExchanger(DuetTokenExchanger):
    def run(self, duet_token: str) -> str:
        self.duet_token = duet_token
        if self.join:
            self._client_exchange(duet_token=self.duet_token)
            return self.responder_id
        else:
            return self._server_exchange(duet_token=self.duet_token)

    def _server_exchange(self, duet_token: str) -> str:
        # send Server ID
        print("♫♫♫ > Duet Server ID: " + bcolors.BOLD + duet_token + bcolors.ENDC)

        print()
        print(
            "♫♫♫ > "
            + bcolors.HEADER
            + "STEP 1:"
            + bcolors.ENDC
            + " Send the following code to your Duet Partner!"
        )
        print("\nimport syft as sy")
        print('duet = sy.duet("' + bcolors.BOLD + duet_token + bcolors.ENDC + '")')

        # get Client ID
        print(
            "\n♫♫♫ > "
            + bcolors.HEADER
            + "STEP 2:"
            + bcolors.ENDC
            + " Have your duet partner send their Client ID to you and enter it below!"
        )
        while True:
            client_id = input("♫♫♫ > Duet Partner's Client ID: ")  # nosec
            if len(client_id) == 32:
                break
            else:
                print("    > Error: Invalid Client ID. Please try again.")
        print()
        return client_id

    def _client_exchange(self, duet_token: str) -> None:
        # send client ID
        print(
            "♫♫♫ > "
            + bcolors.HEADER
            + "STEP 1:"
            + bcolors.ENDC
            + " Send the following Duet Client ID to your duet partner!"
        )
        print("♫♫♫ > Duet Client ID: " + bcolors.BOLD + duet_token + bcolors.ENDC)
        print()
        print("♫♫♫ > ...waiting for partner to connect...")


class OpenGridTokenFileExchanger(DuetTokenExchanger):
    def __init__(
        self,
        *args: TypeTuple[TypeAny, ...],
        file_path: TypeOptional[str] = None,
        **kwargs: TypeAny
    ) -> None:
        super().__init__()
        if file_path is not None:
            self.file_path = file_path
        else:
            self.file_path = OpenGridTokenFileExchanger.get_loopback_path()

    def run(self, duet_token: str) -> str:
        self.duet_token = duet_token
        if self.join:
            return self._client_exchange(duet_token=self.duet_token)
        else:
            return self._server_exchange(duet_token=self.duet_token)

    # for local debugging
    @staticmethod
    def get_loopback_path() -> str:
        loopback_file = "duet_loopback.json"
        return str(Path(tempfile.gettempdir()) / loopback_file)

    def _server_exchange(self, duet_token: str) -> str:
        print()
        print(
            "♫♫♫ > "
            + bcolors.HEADER
            + "STEP 1:"
            + bcolors.ENDC
            + " Send the following code to your Duet Partner!"
        )
        print("\nimport syft as sy")
        print("duet = sy.join_duet(loopback=True)")
        print()

        # send Server ID
        loopback_config = {}
        loopback_config["server_id"] = duet_token
        with open(self.file_path, "w") as f:
            f.write(json.dumps(loopback_config))

        # get Client ID
        client_id = ""
        while client_id == "":
            try:
                with open(self.file_path, "r") as f:
                    loopback_config = json.loads(f.read())
                    if "client_id" in loopback_config:
                        client_id = str(loopback_config["client_id"])
                    else:
                        time.sleep(0.01)
            except Exception as e:
                print(e)
                break
        return client_id

    def _client_exchange(self, duet_token: str) -> str:
        loopback_config = {}
        server_id = ""
        while server_id == "":
            try:
                with open(self.file_path, "r") as f:
                    loopback_config = json.loads(f.read())
                    # only continue once the server has overwritten the file
                    # with only its new server_id
                    if (
                        "server_id" in loopback_config
                        and "client_id" not in loopback_config
                    ):
                        server_id = str(loopback_config["server_id"])
                    else:
                        time.sleep(0.01)
            except Exception as e:
                print(e)
                break

        loopback_config["client_id"] = duet_token

        with open(self.file_path, "w") as f:
            f.write(json.dumps(loopback_config))

        return server_id
