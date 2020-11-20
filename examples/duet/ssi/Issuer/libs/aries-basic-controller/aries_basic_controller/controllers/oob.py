from .base import BaseController
from aiohttp import ClientSession
import logging

logger = logging.getLogger("aries_controller.out-of-band")

class OOBController(BaseController):

    def __init__(self, admin_url: str, client_session: ClientSession):
        super().__init__(admin_url, client_session)


    def default_handler(self, payload):
        logger.debug("Out of Band ", payload)

    async def create_invitation(self, data):
        response = await self.admin_POST(f"/out-of-band/create-invitation", json_data = data)
        return response

    async def receive_invitation(self, data):
        response = await self.admin_POST(f"/out-of-band/receive-invitation", json_data = data)
        return response