from .base import BaseController
from aiohttp import ClientSession
import logging

logger = logging.getLogger("aries_controller.ledger")

class LedgerController(BaseController):

    def __init__(self, admin_url: str, client_session: ClientSession):
        super().__init__(admin_url, client_session)
        self.base_url = "/ledger"

    async def register_nym(self, did, verkey, role: str = None, alias: str = None):
        params = {
            "did": did,
            "verkey": verkey
        }
        if role:
            params["role"] = role
        if alias:
            params["alias"] = alias

        return await self.admin_POST(f"{self.base_url}/register-nym", params=params)

    async def get_nym_role(self, did):
        params = {
            "did": did
        }
        return await self.admin_GET(f"{self.base_url}/get-nym-role", params=params)


    async def get_did_verkey(self, did):
        params = {
            "did": did
        }
        return await self.admin_GET(f"{self.base_url}/did-verkey", params=params)

    async def get_did_endpoint(self, did):
        params = {
            "did": did
        }
        return await self.admin_GET(f"{self.base_url}/did-endpoint", params=params)

    async def get_taa(self):
        return await self.admin_GET(f"{self.base_url}/taa")

    async def accept_taa(self, data):
        return await self.admin_POST(f"{self.base_url}/taa/accept", json_data=data)

    #TODO PATCH rotate key pair


