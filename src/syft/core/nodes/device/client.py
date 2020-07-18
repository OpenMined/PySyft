from ..abstract.client import Client
from typing import final


@final
class DeviceClient(Client):
    def __init__(self, device_id, name, connection):
        super().__init__(worker_id=device_id, name=name, connection=connection)

    def __repr__(self):
        out = f"<DeviceClient id:{self.id}>"
        return out
