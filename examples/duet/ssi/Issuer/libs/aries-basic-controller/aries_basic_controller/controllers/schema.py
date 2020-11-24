from .base import BaseController
from aiohttp import ClientSession
import logging

logger = logging.getLogger("aries_controller.schema")


class SchemaController(BaseController):

    def __init__(self, admin_url: str, client_session: ClientSession):
        super().__init__(admin_url, client_session)
        self.base_url = "/schemas"

    async def get_by_id(self, schema_id):

        response = await self.admin_GET(f"{self.base_url}/{schema_id}")
        return response

    async def get_created_schema(self, schema_id=None, schema_issuer_did=None, schema_name=None, schema_version=None):

        params = {}
        if schema_id:
            params["schema_id"] = schema_id
        if schema_issuer_did:
            params["schema_issuer_did"] = schema_issuer_did
        if schema_name:
            params["schema_name"] = schema_name
        if schema_version:
            params["schema_version"] = schema_version

        response = await self.admin_GET(f"{self.base_url}/created", params=params)

        return response

    async def write_schema(self, schema_name, attributes, schema_version):

        schema_body = {
            "schema_name": schema_name,
            "attributes": attributes,
            "schema_version": schema_version
        }

        response = await self.admin_POST(f"{self.base_url}", schema_body)
        return response






