from ..abstract.client import Client
from typing import final


@final
class DeviceClient(Client):
    def __init__(self, device_id, connection):
        super().__init__(worker_id=device_id, connection=connection)

    def __repr__(self):
        out = f"<DeviceClient id:{self.id}>"
        return out
