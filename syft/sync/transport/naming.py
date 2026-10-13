"""Syft's naming rules for stored files and documents.

These rules belong to syft, not to any backend: a backend stores names, and the
sync layer gives them meaning. They move here from the Drive transport one at a
time.
"""

EVENTS_MESSAGE_PREFIX = "syfteventsmessagev3_"
PROPOSALS_MESSAGE_PREFIX = "msgv2_"

# SYFT_peers.json is a flat map of peer email to entry, so a version at the top
# level would look like a peer email. The version goes under this reserved key.
# A client written before the key reads a peer state from that entry and fails.
# The key therefore never appears as a peer.
PEERS_META_KEY = "_meta"
# Shape of one entry in SYFT_peers.json. Raise this when an entry changes. A file
# with no reserved entry was written before the version, and is version 0.
SYFT_PEERS_VERSION = 1


def timestamp_from_message_name(name: str) -> float | None:
    """The send time a message file carries in its name, or None.

    Both message formats put the time second: ``syfteventsmessagev3_<ts>_<id>``
    for events and ``msgv2_<ts>_<id>`` for proposals.
    """
    if not name.startswith((EVENTS_MESSAGE_PREFIX, PROPOSALS_MESSAGE_PREFIX)):
        return None
    parts = name.split("_")
    if len(parts) < 2:
        return None
    try:
        return float(parts[1])
    except ValueError:
        return None
