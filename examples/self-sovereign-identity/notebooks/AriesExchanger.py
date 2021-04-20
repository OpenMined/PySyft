# stdlib
import asyncio
import json
import time
from typing import Dict as TypeDict
from typing import Optional

# third party
from aries_basic_controller.aries_controller import AriesAgentController

# syft absolute
from syft.grid.duet.exchange_ids import DuetCredentialExchanger

# get_ipython().run_line_magic("autoawait", "")


class AriesDuetTokenExchanger(DuetCredentialExchanger):
    def __init__(self, agent_controller: AriesAgentController) -> None:
        super().__init__()
        self.agent_controller: AriesAgentController = agent_controller
        self.responder_id: Optional[asyncio.Future] = None
        self.proof_request: Optional[TypeDict] = None
        self.is_verified: Optional[asyncio.Future] = None
        self.duet_didcomm_connection_id: Optional[str] = None
        self._register_agent_listeners()

    # The DuetCredentialExchanger expects this method to be implemented.
    # In this case we are establishing a DIDComm connection, challenging the connection
    # with an optional authentication policy, then with successful connections, sending
    # the duet token identifier over this channel.
    def run(
        self,
        credential: str,
    ) -> str:
        self.responder_id = asyncio.Future()
        self.duet_token = credential
        if self.join:
            self._accept_duet_didcomm_invite()
        else:
            self._create_duet_didcomm_invitation()
        loop = asyncio.get_event_loop()

        if self.duet_didcomm_connection_id is not None:
            self.await_active(self.duet_didcomm_connection_id)
        else:
            print("duet_didcomm_connection_id not set")

        print("Sending Duet Token", self.duet_didcomm_connection_id, credential)
        if self.is_verified:
            if self.is_verified.result() is True:
                print("Connection is Verified")
                loop.run_until_complete(
                    self.agent_controller.messaging.send_message(
                        self.duet_didcomm_connection_id, credential
                    )
                )
            else:
                print("Proof request not verified")
        else:
            print("No Proof Requested")
            loop.run_until_complete(
                self.agent_controller.messaging.send_message(
                    self.duet_didcomm_connection_id, credential
                )
            )

        loop.run_until_complete(self.responder_id)

        token = self.responder_id.result()
        print("TOKEN ", token)

        return token

    def _accept_duet_didcomm_invite(self) -> None:

        while True:
            invite = input("♫♫♫ > Duet Partner's Aries Invitation: ")  # nosec
            loop = asyncio.get_event_loop()
            # is_ready = False
            try:
                response = loop.run_until_complete(
                    self.agent_controller.connections.accept_connection(invite)
                )
                print(response["connection_id"])
                connection_id = response["connection_id"]
            except Exception:
                print("    > Error: Invalid invitation. Please try again.")
            break

        self.duet_didcomm_connection_id = connection_id

    def _create_duet_didcomm_invitation(self) -> None:
        loop = asyncio.get_event_loop()
        response = loop.run_until_complete(
            self.agent_controller.connections.create_invitation()
        )
        connection_id = response["connection_id"]
        invite_message = json.dumps(response["invitation"])

        print()
        print("♫♫♫ > " + "STEP 1:" + " Send the aries invitation to your Duet Partner!")
        print()
        print(invite_message)
        print()
        self.duet_didcomm_connection_id = connection_id

    # Should be converted to asycio Future
    def await_active(self, connection_id: str) -> None:
        print("Waiting for active connection", connection_id)

        while True:
            loop = asyncio.get_event_loop()
            response = loop.run_until_complete(
                self.agent_controller.connections.get_connection(connection_id)
            )
            is_ready = "active" == response["state"]
            if is_ready:
                print("Connection Active")
                if self.proof_request:
                    self.is_verified = asyncio.Future()
                    self.challenge_connection(connection_id)
                    loop.run_until_complete(self.is_verified)
                break
            else:
                time.sleep(2)

    def challenge_connection(self, connection_id: str) -> None:
        loop = asyncio.get_event_loop()
        proof_request_web_request = {
            "connection_id": connection_id,
            "proof_request": self.proof_request,
            "trace": False,
        }
        response = loop.run_until_complete(
            self.agent_controller.proofs.send_request(proof_request_web_request)
        )
        print("Challenge")
        print(response)
        pres_ex_id = response["presentation_exchange_id"]
        print(pres_ex_id)

    def _register_agent_listeners(self) -> None:
        print("REGISTER LISTENERS")
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.agent_controller.listen_webhooks())

        listeners = [
            {"handler": self.messages_handler, "topic": "basicmessages"},
            {"topic": "issue_credential", "handler": self.cred_handler},
            {"handler": self.connection_handler, "topic": "connections"},
            {"topic": "present_proof", "handler": self.proof_handler},
        ]
        self.agent_controller.register_listeners(listeners, defaults=True)

    def cred_handler(self, payload: TypeDict) -> None:
        print("Handle Credentials")
        exchange_id = payload["credential_exchange_id"]
        state = payload["state"]
        role = payload["role"]
        attributes = payload["credential_proposal_dict"]["credential_proposal"][
            "attributes"
        ]
        print(f"Credential exchange {exchange_id}, role: {role}, state: {state}")
        print(f"Offering: {attributes}")

    def connection_handler(self, payload: TypeDict) -> None:
        print("Connection Handler Called")
        connection_id = payload["connection_id"]
        state = payload["state"]
        print(f"Connection {connection_id} in State {state}")

    def proof_handler(self, payload: TypeDict) -> None:
        print("Handle present proof")
        role = payload["role"]
        pres_ex_id = payload["presentation_exchange_id"]
        state = payload["state"]
        print(f"Role {role}, Exchange {pres_ex_id} in state {state}")
        loop = asyncio.get_event_loop()

        if state == "presentation_received":
            verify = loop.run_until_complete(
                self.agent_controller.proofs.verify_presentation(pres_ex_id)
            )
            if self.is_verified is not None:
                self.is_verified.set_result(verify["state"] == "verified")
            else:
                print("is_verified Future has not been created")

    # Receive basic messages
    def messages_handler(self, payload: TypeDict) -> None:
        print("Handle Duet ID", payload["content"])
        if self.responder_id is not None:
            self.responder_id.set_result(payload["content"])
        else:
            print("responder_id Future has not been created")

    # Used for other Aries connections. E.g. with an issuer
    def accept_connection(self, invitation: str) -> str:
        # Receive Invitation
        loop = asyncio.get_event_loop()
        response = loop.run_until_complete(
            self.agent_controller.connections.accept_connection(invitation)
        )
        # Print out accepted Invite and Alice's connection ID
        print("Connection", response)
        return response["connection_id"]

    def create_invitation(self) -> str:
        # Create Invitation
        loop = asyncio.get_event_loop()
        invite = loop.run_until_complete(
            self.agent_controller.connections.create_invitation()
        )
        # connection_id = invite["connection_id"]
        invite_message = json.dumps(invite["invitation"])
        return invite_message

    def configure_challenge(self, proof_request: TypeDict) -> None:
        self.proof_request = proof_request
