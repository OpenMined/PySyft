from .base import BaseController
from aiohttp import ClientSession
import logging
from typing import List

logger = logging.getLogger("aries_controller.proof")

PRES_PREVIEW = "did:sov:BzCbsNYhMrjHiqZDTUASHg;spec/present-proof/1.0/presentation-preview"

class ProofController(BaseController):

    def __init__(self, admin_url: str, client_session: ClientSession):
        super().__init__(admin_url, client_session)
        self.base_url = "/present-proof"

    def default_handler(self, payload):
        logger.debug("Present Proof Message received", payload)

    async def get_records(self, connection_id: str = None, thread_id: str = None, state: str = None, role: str = None):
        params = {}
        if connection_id:
            params["connection_id"] = connection_id
        if thread_id:
            params["thread_id"] = thread_id
        if state:
            params["state"] = state
        if role:
            params["role"] = role

        return await self.admin_GET(f"{self.base_url}/records", params=params)

    async def get_record_by_id(self, pres_ex_id):
        return await self.admin_GET(f"{self.base_url}/records/{pres_ex_id}")

    # Fetch a single presentation exchange record
    async def get_presentation_credentials(self, pres_ex_id, count: int = None, wql_query: str = None, start: int = None, referent: str = None):
        params = {}
        if count:
            params["count"] = count
        # Not sure what this does
        if wql_query:
            params["extra_query"] = wql_query
        if start:
            params["start"] = start
        # Not sure what this does
        if referent:
            params["referent"] = referent

        return await self.admin_GET(f"{self.base_url}/records/{pres_ex_id}/credentials", params=params)

    # Sends a presentation proposal
    async def send_proposal(self, proposal):

        return await self.admin_POST(f"{self.base_url}/send-proposal", json_data=proposal)

    # Creates a presentation request not bound to any proposal or existing connection
    async def create_request(self, request):
        # TODO How should proof request object be broken up? Complex.
        #  Do we want user to have to know how to build this object?

        return await self.admin_POST(f"{self.base_url}/create-request", json_data=request)


    # Sends a free presentation request not bound to any proposal
    async def send_request(self, request):
        return await self.admin_POST(f"{self.base_url}/send-request", json_data=request)

    async def send_request_for_proposal(self, pres_ex_id, proposal_request):

        return await self.admin_POST(f"{self.base_url}/records/{pres_ex_id}/send-request", json_data=proposal_request)

    # Send a proof presentation
    async def send_presentation(self, pres_ex_id, presentation):

        return await self.admin_POST(f"{self.base_url}/records/{pres_ex_id}/send-presentation", json_data=presentation)

    # Verify a received presentation
    async def verify_presentation(self, pres_ex_id):
        return await self.admin_POST(f"{self.base_url}/records/{pres_ex_id}/verify-presentation")

    async def remove_presentation_record(self, pres_ex_id):
        return await self.admin_DELETE(f"{self.base_url}/records/{pres_ex_id}")

    # def build_proof_request(self, name, version, requested_attributes, requested_predicates):

