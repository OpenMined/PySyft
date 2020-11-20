from .base import BaseController
from aiohttp import ClientSession
import logging

logger = logging.getLogger("aries_controller.definitions")

class DefinitionsController(BaseController):

    def __init__(self, admin_url: str, client_session: ClientSession):
        super().__init__(admin_url, client_session)
        self.base_url = "/credential-definitions"

    async def get_by_id(self, cred_def_id):
        return await self.admin_GET(f"{self.base_url}/{cred_def_id}")

    async def write_cred_def(self, schema_id, tag: str = "default", support_revocation: bool = False):

        body = {
            "schema_id": schema_id,
            "tag": tag,
            "support_revocation": support_revocation
        }

        return await self.admin_POST(f"{self.base_url}", body)

    async def search_created(self, schema_id = None, schema_issuer_did=None, schema_name=None,
                             schema_version=None, issuer_did=None, cred_def_id=None):

        params = {}
        if schema_id:
            params["schema_id"] = schema_id
        if schema_issuer_did:
            params["schema_issuer_did"] = schema_issuer_did
        if schema_version:
            params["schema_version"] = schema_version
        if schema_name:
            params["schema_name"] = schema_name
        if issuer_did:
            params["issuer_did"] = issuer_did
        if cred_def_id:
            params["cred_def_id"] = cred_def_id

        return await self.admin_GET(f"{self.base_url}/created", params=params)

