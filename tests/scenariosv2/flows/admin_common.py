# syft absolute
import syft as sy

# relative
from ..sim.core import SimulatorContext
from .utils import server_info

__all__ = ["register_user"]


def register_user(ctx: SimulatorContext, admin_client: sy.DatasiteClient, user: dict):
    msg = f"Admin {admin_client.metadata.server_side_type}: User {user['email']} on {server_info(admin_client)}"
    ctx.logger.info(f"{msg} - Creating")
    _ = admin_client.register(
        name=user["name"],
        email=user["email"],
        password=user["password"],
        password_verify=user["password"],
    )
    ctx.logger.info(f"{msg} - Created")
