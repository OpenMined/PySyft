# Peer request flow

1. **DS requests access**: `ds_manager.add_peer(do_email)` - Creates a peer request stored in transport layer
2. **DO loads requests**: `do_manager.load_peers()` - Populates `_peer_requests` list
3. **DO approves**: `do_manager.approve_peer_request(ds_email)` - Moves peer from `_peer_requests` to `_approved_peers`
4. **Sync enabled**: Only approved peers can sync files. Unapproved peer messages are ignored.

Peer states: `ACCEPTED`, `PENDING`, `REJECTED`
