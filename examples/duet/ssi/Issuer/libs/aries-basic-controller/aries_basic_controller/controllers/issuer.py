from .base import BaseController
from aiohttp import ClientSession
import logging
from ..helpers.utils import extract_did, get_schema_details
logger = logging.getLogger("aries_controller.issuer")

CRED_PREVIEW = "did:sov:BzCbsNYhMrjHiqZDTUASHg;spec/issue-credential/1.0/credential-preview"

class IssuerController(BaseController):

    def __init__(self, admin_url: str, client_session: ClientSession, connection_controller,
                 wallet_controller, definition_controller):
        super().__init__(admin_url, client_session)
        self.base_url = "/issue-credential"
        self.connections = connection_controller
        self.wallet = wallet_controller
        self.definitions = definition_controller

    # Fetch all credential exchange records
    async def get_records(self):

        return await self.admin_GET(f"{self.base_url}/records")

    async def get_record_by_id(self, cred_ex_id):
        return await self.admin_GET(f"{self.base_url}/records/{cred_ex_id}")

    # Create a credential, automating the entire flow
    # TODO trace=True causes error. Not sure why
    # async def create_credential(self, connection_id, schema_id, cred_def_id, attributes, comment: str = "",
    #                           auto_remove: bool = True, trace: bool = False):
    #
    #     body = await self.create_credential_body(connection_id, schema_id, cred_def_id, attributes, comment,
    #                                              auto_remove, trace)
    async def create_credential(self, body):
        return await self.admin_POST(f"{self.base_url}/create", json_data=body)

    # Send holder a credential, automating the entire flow
    # TODO trace=True causes error. Not sure why
    async def send_credential(self, connection_id, schema_id, cred_def_id, attributes, comment: str = "",
                              auto_remove: bool = True, trace: bool = False):

        body = await self.create_credential_body(connection_id, schema_id, cred_def_id, attributes, comment,
                                                 auto_remove, trace)
        return await self.admin_POST(f"{self.base_url}/send", json_data=body)


    # Send Issuer a credential proposal
    async def send_proposal(self, connection_id, schema_id, cred_def_id, attributes, comment: str = "",
                            auto_remove: bool = True, trace: bool = False):

        body = await self.create_credential_body(connection_id, schema_id, cred_def_id, attributes, comment,
                                                 auto_remove, trace)
        return await self.admin_POST(f"{self.base_url}/send-proposal", json_data=body)

    async def send_offer(self, connection_id, cred_def_id, attributes, comment: str = "",
                         auto_issue: bool = True, auto_remove: bool = True, trace: bool = False):
        await self.connections.is_active(connection_id)
        offer_body = {
            "cred_def_id": cred_def_id,
            "auto_remove": auto_remove,
            "trace": trace,
            "comment": comment,
            "auto_issue": auto_issue,
            "credential_preview": {
                "@type": CRED_PREVIEW,
                "attributes": attributes
            },
            "connection_id": connection_id
        }
        return await self.admin_POST(f"{self.base_url}/send-offer", json_data=offer_body)

    # Send holder a credential offer in reference to a proposal with preview
    async def send_offer_for_record(self, cred_ex_id):
        return await self.admin_POST(f"{self.base_url}/records/{cred_ex_id}/send-offer")

    # Send issuer a credential request
    async def send_request_for_record(self, cred_ex_id):
        return await self.admin_POST(f"{self.base_url}/records/{cred_ex_id}/send-request")

    # Send holder a credential
    async def issue_credential(self, cred_ex_id, comment, attributes):
        body = {
            "comment": comment,
            "credential_preview": {
                "@type": CRED_PREVIEW,
                "attributes": attributes
            }
        }
        return await self.admin_POST(f"{self.base_url}/records/{cred_ex_id}/issue", json_data=body)

    # Store a received credential
    async def store_credential(self, cred_ex_id, credential_id):
        body = {
            "credential_id": credential_id
        }
        return await self.admin_POST(f"{self.base_url}/records/{cred_ex_id}/store", json_data=body)

    # Revoke and issued credential
    # async def revoke_credential(self, rev_reg_id, cred_rev_id, publish: bool = False):
    #     params = {
    #         "rev_reg_id": rev_reg_id,
    #         "cred_reg_id": cred_rev_id,
    #         "publish": publish
    #     }
    #     return await self.admin_POST(f"{self.base_url}/revoke", params=params)
    #
    # # Publish pending revocations
    # async def publish_revocations(self):
    #     return await self.admin_POST(f"{self.base_url}/publish-revocations")

    # Remove an existing credential exchange record
    async def remove_record(self, cred_ex_id):
        return await self.admin_DELETE(f"{self.base_url}/records/{cred_ex_id}")

    # Send a problem report for a credential exchange
    async def problem_report(self, cred_ex_id, explanation: str):
        body = {
            "explain_ltxt": explanation
        }

        return await self.admin_POST(f"{self.base_url}/records/{cred_ex_id}/problem-report", json_data=body)

    # Used for both send and send-proposal bodies
    async def create_credential_body(self, connection_id, schema_id, cred_def_id, attributes, comment: str = "",
                                     auto_remove: bool = True, trace: bool = False):
        # raises error if connection not active
        await self.connections.is_active(connection_id)

        schema_details = get_schema_details(schema_id)

        issuer_did = extract_did(cred_def_id)

        body = {
            "issuer_did": issuer_did,
            "auto_remove": auto_remove,
            "credential_proposal": {
                "@type": CRED_PREVIEW,
                "attributes": attributes
            },
            "connection_id": connection_id,
            "trace": trace,
            "comment": comment,
            "cred_def_id": cred_def_id,
        }

        credential_body = {**body, **schema_details}
        return credential_body
