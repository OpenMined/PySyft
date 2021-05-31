# stdlib
import asyncio
import logging
from typing import Dict as TypeDict
from typing import List as TypeList
from typing import Optional

# third party
from aries_cloudcontroller import AriesAgentController
import nest_asyncio

nest_asyncio.apply()

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class OMAuthorityAgent:
    def __init__(
        self,
        agent_controller: AriesAgentController,
        data_owner_config: TypeDict,
        data_scientist_config: TypeDict,
    ) -> None:

        self.agent_controller = agent_controller

        self.agent_listeners = [
            {"topic": "connections", "handler": self._connections_handler},
            {"topic": "present_proof", "handler": self._proof_handler},
            {"topic": "issue_credential", "handler": self._cred_handler},
        ]

        self.agent_controller.register_listeners(self.agent_listeners, defaults=True)

        self.trusted_client_connection_ids: TypeList[str] = []
        self.pending_client_connection_ids: TypeList[str] = []
        self.scientist_connection_ids: TypeList[str] = []
        self.datascientist_details_list: TypeList[TypeDict] = []
        self.dataowner_connection_ids: TypeList[str] = []
        self.dataowner_details_list: TypeList[TypeDict] = []

        # JSON objects defining a schema_id, cred_def_id pair.
        self.data_owner_config = data_owner_config
        self.data_scientist_config = data_scientist_config

        self.client_auth_policy: Optional[TypeDict] = None

    def _cred_handler(self, payload: TypeDict) -> None:
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
        if state == "offer_sent":
            proposal = payload["credential_proposal_dict"]
            attributes = proposal["credential_proposal"]["attributes"]

            print(f"Offering credential with attributes  : {attributes}")
            # YOUR LOGIC HERE
        elif state == "request_received":
            print("Request for credential received")
            # YOUR LOGIC HERE
        elif state == "credential_sent":
            print("Credential Sent")
            # YOUR LOGIC HERE

    def _connections_handler(self, payload: TypeDict) -> None:
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
        # if state == "response":
        #
        #     # Ensures connections moved to active
        #     loop = asyncio.get_event_loop()
        #     loop.create_task(self.agent_controller.messaging.trust_ping(connection_id, 'hello!'))
        #     time.sleep(3)
        #     loop.create_task(self.agent_controller.messaging.trust_ping(connection_id, 'hello!'))
        if state == "active":
            loop = asyncio.get_event_loop()
            if connection_id in self.pending_client_connection_ids:
                if self.client_auth_policy:

                    # Specify the connection id to send the authentication request to
                    proof_request_web_request = {
                        "connection_id": connection_id,
                        "proof_request": self.client_auth_policy,
                        "trace": False,
                    }
                    _ = loop.run_until_complete(
                        self.agent_controller.proofs.send_request(
                            proof_request_web_request
                        )
                    )
            elif connection_id in self.scientist_connection_ids:
                for datascientist in self.datascientist_details_list:
                    if datascientist["connection_id"] == connection_id:
                        schema_id = self.data_scientist_config["schema_id"]
                        cred_def_id = self.data_scientist_config["cred_def_id"]

                        # NOTE IF YOU CHANGE THE SCHEMA YOU ARE ISSUING AGAINST THIS WILL NEED TO BE UPDATED
                        credential_attributes = [
                            {"name": "name", "value": datascientist["name"]},
                            {"name": "scope", "value": datascientist["scope"]},
                        ]
                        log_msg = (
                            f"issuing data scientist - {connection_id} a credential "
                            + f"using schema {schema_id} and definition {cred_def_id}"
                        )
                        logger.info(log_msg)
                        loop.run_until_complete(
                            self.agent_controller.issuer.send_credential(
                                connection_id,
                                schema_id,
                                cred_def_id,
                                credential_attributes,
                                trace=False,
                            )
                        )
                        break
            elif connection_id in self.dataowner_connection_ids:
                for dataowner in self.dataowner_details_list:
                    if dataowner["connection_id"] == connection_id:
                        schema_id = self.data_owner_config["schema_id"]
                        cred_def_id = self.data_owner_config["cred_def_id"]
                        credential_attributes = [
                            {"name": "name", "value": dataowner["name"]},
                            {"name": "domain", "value": dataowner["domain"]},
                        ]
                        log_msg = (
                            f"issuing data scientist - {connection_id} a credential "
                            + f"using schema {schema_id} and definition {cred_def_id}"
                        )
                        logger.info(log_msg)
                        loop.run_until_complete(
                            self.agent_controller.issuer.send_credential(
                                connection_id,
                                schema_id,
                                cred_def_id,
                                credential_attributes,
                                trace=False,
                            )
                        )

    def _proof_handler(self, payload: TypeDict) -> None:
        role = payload["role"]
        connection_id = payload["connection_id"]
        pres_ex_id = payload["presentation_exchange_id"]
        state = payload["state"]
        print(
            "\n---------------------------------------------------------------------\n"
        )
        print("Handle present-proof")
        print("Connection ID : ", connection_id)
        print("Presentation Exchange ID : ", pres_ex_id)
        print("Protocol State : ", state)
        print("Agent Role : ", role)
        print(
            "\n---------------------------------------------------------------------\n"
        )
        if state == "presentation_received":
            # Only verify presentation's from pending scientist connections
            if connection_id in self.pending_client_connection_ids:

                print("Connection is a pending scientist")

                loop = asyncio.get_event_loop()
                print("Verifying Presentation from Data Scientist")
                verify = loop.run_until_complete(
                    self.agent_controller.proofs.verify_presentation(pres_ex_id)
                )
                # Completing future with result of the verification - True of False
                if verify["state"] == "verified":
                    self.trusted_client_connection_ids.append(connection_id)
                self.pending_client_connection_ids.remove(connection_id)

    def client_invitation(self) -> TypeDict:

        loop = asyncio.get_event_loop()

        client_invite = loop.run_until_complete(
            self.agent_controller.connections.create_invitation()
        )

        connection_id = client_invite["connection_id"]

        self.pending_client_connection_ids.append(connection_id)

        return {
            "invite_url": client_invite["invitation_url"],
            "connection_id": connection_id,
        }

    def data_scientist_invitation(self, name: str, scope: str) -> TypeDict:
        loop = asyncio.get_event_loop()

        scientist_invite = loop.run_until_complete(
            self.agent_controller.connections.create_invitation()
        )

        connection_id = scientist_invite["connection_id"]

        self.scientist_connection_ids.append(connection_id)

        datascientist = {"connection_id": connection_id, "scope": scope, "name": name}
        self.datascientist_details_list.append(datascientist)

        return scientist_invite["invitation"]

    def data_owner_invitation(self, name: str, domain: str) -> TypeDict:
        loop = asyncio.get_event_loop()

        owner_invite = loop.run_until_complete(
            self.agent_controller.connections.create_invitation()
        )

        connection_id = owner_invite["connection_id"]

        self.dataowner_connection_ids.append(connection_id)
        dataowner = {"connection_id": connection_id, "domain": domain, "name": name}
        self.dataowner_details_list.append(dataowner)
        return owner_invite["invitation"]

    def client_connection_trusted(self, connection_id: str) -> bool:
        return connection_id in self.trusted_client_connection_ids

    def set_client_auth_policy(self, proof_request: TypeDict) -> None:
        self.client_auth_policy = proof_request
