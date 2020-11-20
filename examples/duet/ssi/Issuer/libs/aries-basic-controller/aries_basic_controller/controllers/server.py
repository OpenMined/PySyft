from .base import BaseController
from aiohttp import ClientSession
import logging

logger = logging.getLogger("aries_controller.proof")


class ServerController(BaseController):

    def __init__(self, admin_url: str, client_session: ClientSession):
        super().__init__(admin_url, client_session)

    async def get_plugins(self):
        return await self.admin_GET('/plugins')

    async def get_status(self):
        return await self.admin_GET('/status')

    async def reset_status(self):
        return await self.admin_POST('/status/reset')

    async def status_ready(self):
        return await self.admin_GET('/status/ready')

    async def status_live(self):
        return await self.admin_GET('/status/live')

    async def get_features(self, query: str = None):
        params = {}
        if query:
            params["query"] = query

        return await self.admin_GET('/features', params=params)

    ## TODO implement shutdown. Should this be a POST?