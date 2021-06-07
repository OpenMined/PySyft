# stdlib
import asyncio
import json
import time
from typing import Dict as TypeDict
from typing import Optional

# third party
from aries_cloudcontroller import AriesAgentController

# syft absolute
from syft.grid.duet.exchange_ids import DuetCredentialExchanger


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
                    self.agent_controller.connections.receive_invitation(invite)
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

        listeners = [
            {"handler": self.messages_handler, "topic": "basicmessages"},
            {"topic": "issue_credential", "handler": self.cred_handler},
            {"handler": self.connection_handler, "topic": "connections"},
            {"topic": "present_proof", "handler": self.proof_handler},
        ]
        self.agent_controller.register_listeners(listeners, defaults=True)

    def cred_handler(self, payload: TypeDict) -> None:
        connection_id = payload["connection_id"]
        exchange_id = payload["credential_exchange_id"]
        state = payload["state"]
        role = payload["role"]
        print("\n---------------------------------------------------\n")
        print("Handle Issue Credential Webhook")
        print(f"Connection ID : {connection_id}")
        print(f"Credential exchange ID : {exchange_id}")
        print("Agent Protocol Role : ", role)
        print("Protocol State : ", state)
        print("\n---------------------------------------------------\n")
        print("Handle Credential Webhook Payload")

        if state == "offer_received":
            print("Credential Offer Recieved")
            proposal = payload["credential_proposal_dict"]
            print(
                "The proposal dictionary is likely how you would understand and "
                + "display a credential offer in your application"
            )
            print("\n", proposal)
            print("\n This includes the set of attributes you are being offered")
            attributes = proposal["credential_proposal"]["attributes"]
            print(attributes)
            # YOUR LOGIC HERE
        elif state == "request_sent":
            print(
                "\nA credential request object contains the commitment to the agents "
                + "master secret using the nonce from the offer"
            )
            # YOUR LOGIC HERE
        elif state == "credential_received":
            print("Received Credential")
            # YOUR LOGIC HERE
        elif state == "credential_acked":
            # YOUR LOGIC HERE
            credential = payload["credential"]
            print("Credential Stored\n")
            print(credential)

            print(
                "\nThe referent acts as the identifier for retrieving the raw credential from the wallet"
            )
            # Note: You would probably save this in your application database
            credential_referent = credential["referent"]
            print("Referent", credential_referent)

    def connection_handler(self, payload: TypeDict) -> None:
        state = payload["state"]
        connection_id = payload["connection_id"]
        their_role = payload["their_role"]
        routing_state = payload["routing_state"]

        print("----------------------------------------------------------")
        print("Connection Webhook Event Received")
        print("Connection ID : ", connection_id)
        print("State : ", state)
        print("Routing State : ", routing_state)
        print("Their Role : ", their_role)
        print("----------------------------------------------------------")

    def proof_handler(self, payload: TypeDict) -> None:
        role = payload["role"]
        connection_id = payload["connection_id"]
        pres_ex_id = payload["presentation_exchange_id"]
        state = payload["state"]
        loop = asyncio.get_event_loop()
        print(
            "\n---------------------------------------------------------------------\n"
        )
        print("Handle present-proof")
        print("Connection ID : ", connection_id)
        print("Presentation Exchange ID : ", pres_ex_id)
        print("Protocol State : ", state)
        print("Agent Role : ", role)
        print("Initiator : ", payload["initiator"])
        print(
            "\n---------------------------------------------------------------------\n"
        )

        if state == "presentation_received":
            verified_response = loop.run_until_complete(
                self.agent_controller.proofs.verify_presentation(pres_ex_id)
            )
            if self.is_verified is not None:
                self.is_verified.set_result(verified_response["verified"] == "true")
                print("Attributes Presented")
                for (name, val) in verified_response["presentation"]["requested_proof"][
                    "revealed_attrs"
                ].items():
                    # This is the actual data that you want. It's a little hidden
                    print("Attribute : ", val)
                    print("Raw Value : ", val["raw"])
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
    def receive_invitation(self, invitation: str) -> str:
        # Receive Invitation
        loop = asyncio.get_event_loop()
        response = loop.run_until_complete(
            self.agent_controller.connections.receive_invitation(invitation)
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
