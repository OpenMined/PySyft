# stdlib
from typing import Any


# Decryption shouldn't be possible over Duet pointers.
# We retrieve the data, link it to the secret context, and then decrypt.
def decrypt(secret_ctx: Any, ptr: Any) -> Any:
    local = ptr.get()
    local.link_context(secret_ctx)

    return local.decrypt()
