# stdlib
import json
from pathlib import Path
import tempfile
import time
from typing import Any as TypeAny
from typing import Optional as TypeOptional
from typing import Tuple as TypeTuple

# syft relative
from ...logger import debug
from ...logger import info
from ...logger import traceback_and_raise
from .bcolors import bcolors


class DuetCredentialExchanger:
    def __init__(self, *args: TypeTuple[TypeAny, ...], **kwargs: TypeAny) -> None:
        self.join = False

    def set_role(self, join: bool = False) -> "DuetCredentialExchanger":
        self.join = join
        return self

    def set_responder_id(self, credential: TypeAny) -> "DuetCredentialExchanger":
        self.responder_id = credential
        return self

    def run(self, credential: str) -> str:
        traceback_and_raise(NotImplementedError)


class OpenGridTokenManualInputExchanger(DuetCredentialExchanger):
    def run(self, credential: str) -> str:
        self.credential = credential
        if self.join:
            self._client_exchange(credential=self.credential)
            return self.responder_id
        else:
            return self._server_exchange(credential=self.credential)

    def _server_exchange(self, credential: str) -> str:
        # send Server ID
        info(
            "♫♫♫ > Duet Server ID: " + bcolors.BOLD + credential + bcolors.ENDC,
            print=True,
        )

        info(print=True)
        info(
            "♫♫♫ > "
            + bcolors.HEADER
            + "STEP 1:"
            + bcolors.ENDC
            + " Send the following code to your Duet Partner!",
            print=True,
        )
        info("\nimport syft as sy", print=True)
        info(
            'duet = sy.duet("' + bcolors.BOLD + credential + bcolors.ENDC + '")',
            print=True,
        )

        # get Client ID
        info(
            "\n♫♫♫ > "
            + bcolors.HEADER
            + "STEP 2:"
            + bcolors.ENDC
            + " Ask your partner for their Client ID and enter it below!",
            print=True,
        )
        while True:
            client_id = input("♫♫♫ > Duet Partner's Client ID: ")  # nosec
            if len(client_id) == 32:
                break
            else:
                info("    > Error: Invalid Client ID. Please try again.", print=True)
        info(print=True)
        return client_id

    def _client_exchange(self, credential: str) -> None:
        # send client ID
        info(print=True)
        info(
            "♫♫♫ > "
            + bcolors.HEADER
            + "STEP 1:"
            + bcolors.ENDC
            + " Send the following Duet Client ID to your duet partner!",
            print=True,
        )
        info(
            "♫♫♫ > Duet Client ID: " + bcolors.BOLD + credential + bcolors.ENDC,
            print=True,
        )
        info(print=True)
        info("♫♫♫ > ...waiting for partner to connect...", print=True)


def get_loopback_path() -> str:
    loopback_file = "duet_loopback.json"
    return str(Path(tempfile.gettempdir()) / loopback_file)


class OpenGridTokenFileExchanger(DuetCredentialExchanger):
    retry_seconds = 180
    file_path = get_loopback_path()

    def __init__(
        self,
        *args: TypeTuple[TypeAny, ...],
        file_path: TypeOptional[str] = None,
        **kwargs: TypeAny
    ) -> None:
        super().__init__()
        if file_path is not None:
            self.file_path = file_path

    def run(self, credential: str) -> str:
        self.credential = credential
        if self.join:
            return self._client_exchange(credential=self.credential)
        else:
            return self._server_exchange(credential=self.credential)

    def _server_exchange(self, credential: str) -> str:
        info(print=True)
        info(
            "♫♫♫ > "
            + bcolors.HEADER
            + "STEP 1:"
            + bcolors.ENDC
            + " Send the following code to your Duet Partner!",
            print=True,
        )
        info("\nimport syft as sy", print=True)
        info("duet = sy.join_duet(loopback=True)", print=True)
        info(print=True)

        # send Server ID
        loopback_config = {}
        loopback_config["server_id"] = credential
        with open(self.file_path, "w") as f:
            f.write(json.dumps(loopback_config))

        # get Client ID
        client_id = ""
        for retry in range(OpenGridTokenFileExchanger.retry_seconds):
            try:
                with open(self.file_path, "r") as f:
                    loopback_config = json.loads(f.read())

                    if "client_id" in loopback_config:
                        client_id = str(loopback_config["client_id"])
                        break
                debug("client not ready")
                time.sleep(0.5)
            except KeyboardInterrupt:
                debug("Cancelling server connection")
                break
            except Exception as e:
                info("server config load failed", self.file_path, e)
                time.sleep(0.5)

        if client_id == "":
            raise Exception("failed to load client ID")

        return client_id

    def _client_exchange(self, credential: str) -> str:
        loopback_config = {}
        server_id = ""
        for retry in range(OpenGridTokenFileExchanger.retry_seconds):
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
                        break
                debug("server not ready")
                time.sleep(0.5)
            except KeyboardInterrupt:
                debug("Cancelling client connection")
                break
            except Exception as e:
                info("client config load failed", self.file_path, e)
                time.sleep(0.5)

        if server_id == "":
            raise Exception("failed to load client ID")

        loopback_config["client_id"] = credential

        with open(self.file_path, "w") as f:
            f.write(json.dumps(loopback_config))

        return server_id
