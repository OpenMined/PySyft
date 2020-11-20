from .base import BaseController
from aiohttp import ClientSession
import logging

logger = logging.getLogger("aries_controller.credentials")

class CredentialController(BaseController):

    def __init__(self, admin_url: str, client_session: ClientSession):
        super().__init__(admin_url, client_session)

    async def get_by_id(self, credential_id):
        return await self.admin_GET(f"/credential/{credential_id}")

    async def get_credential_mime_types(self, credential_id):
        return await self.admin_GET(f"/credential/mime-types/{credential_id}")

    async def remove_credential(self, credential_id):
        return await self.admin_DELETE(f"/credential/{credential_id}")

    async def get_all(self, wql_query: str = None, count: int = None, start: int = None):
        params = {}
        if wql_query:
            params["wql"] = wql_query
        if count:
            params["count"] = count
        if start:
            params["start"] = start

        return await self.admin_GET("/credentials", params=params)

    async def is_revoked(self, credential_id):
        return await self.admin_GET(f"credential/revoked/{credential_id}")