# syft absolute
import syft as sy


def server_info(client: sy.DatasiteClient) -> str:
    url = getattr(client.connection, "url", "python")
    return f"{client.name}(url={url}, side={client.metadata.server_side_type})"
