"""Wipe all SyftBox state (Drive + local) for the given accounts.

Run before a demo so every datasite starts clean and re-publishes a fresh
version file — this avoids the peer-version handshake race where the enclave
skips a data owner whose version file was deleted but not yet re-shared.

Invoked by ``just inference-reset``. Usage::

    python reset_state.py EMAIL=TOKEN [EMAIL=TOKEN ...]
"""

import os
import sys

os.environ["PRE_SYNC"] = "false"

from syft_enclaves import login_do

for arg in sys.argv[1:]:
    email, _, token = arg.partition("=")
    client = login_do(email, token, sync=False, load_peers=False)
    # broadcast_delete_events=False: a full reset doesn't need (and shouldn't
    # depend on) reading peer versions, which is exactly what we're resetting.
    client.delete_syftbox(broadcast_delete_events=False)
    print(f"wiped {email}")
