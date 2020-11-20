from .base import BaseController
from ..models.connection import Connection

from aiohttp import (
    ClientSession,
)
import logging
import asyncio

class ActionMenuController(BaseController):

    def __init__(self, admin_url: str, client_session: ClientSession):
        super().__init__(admin_url, client_session)

    async def close_menu(self, connection_id: str):
        return await self.admin_POST(f"/action-menu/{connection_id}/close")

    async def get_menu(self, connection_id: str):
        return await self.admin_POST(f"/action-menu/{connection_id}/fetch")

    async def request_menu(self, connection_id: str):
        return await self.admin_POST(f"/action-menu/{connection_id}/request")

    async def perform(self, connection_id: str, menu_params,  menu_option_name):
        body = {
            "params": menu_params,
            "name": menu_option_name
        }

        return await self.admin_POST(f"/action-menu/{connection_id}/perform", json_data=body)

    async def send_menu(
        self,
        connection_id: str,
        menu_description: str,
        menu_errormsg: str,
        menu_title: str,
        menu_options,
    ):
        body = {
            "menu": {
                "description": menu_description,
                "errormsg": menu_errormsg,
                "title": menu_title,
                "options": menu_options
            }
        }

        return await self.admin_POST(f"/connections/{connection_id}/send-menu", json_data=body)