from aries_basic_controller.controllers.base import BaseController
from aries_basic_controller.controllers.connections import ConnectionsController

from aiohttp import ClientSession

# This is an example for how we might extend the basic controller
class ProtocolController(BaseController):

    def __init__(self, admin_url: str, client_session: ClientSession, connections_controller: ConnectionsController):
        super().__init__(admin_url, client_session)
        self.connections = connections_controller

    async def send_attachment(self, connection_id, data):

        await self.connections.is_active(connection_id)

        path = f"/connections/{connection_id}/send-attachment"
        return await self.admin_POST(path, data=data)

 # async def store_attachment(self,data):

 
