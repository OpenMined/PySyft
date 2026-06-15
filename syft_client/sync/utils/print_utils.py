from syft_client.sync.peers.peer import Peer
from syft_client.sync.utils.syftbox_utils import check_env
from syft_client.sync.environments.environment import Environment
from syft_client.version import SYFT_CLIENT_VERSION
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from syft_client.sync.syftbox_manager import SyftboxManager


def print_client_connecting(email: str):
    is_colab = check_env() == Environment.COLAB
    print(f"🔑 Logging in as {email}...")
    print(f"   Environment: {'Colab' if is_colab else 'Python'}")


def print_client_connected(client: "SyftboxManager"):
    print("\n✅ Logged in successfully!")
    print(f"   SyftBox folder : {client.syftbox_folder}")
    print(f"   Version        : {SYFT_CLIENT_VERSION}")

    peers = client.peer_manager.approved_peers
    if peers:
        print(f"\n👥 {len(peers)} peer(s) restored from previous session.")
        print("   Run client.sync() to pull the latest updates.")
    else:
        print("\n💡 No peers yet. Add a Data Owner with:")
        print("   client.add_peer('do@org.com')")


def print_peer_adding_to_platform(peer_email: str, platform_str: str):
    print(f"🔄 Adding {peer_email} on {platform_str}...")


def print_peer_added_to_platform(peer_email: str, platform_str: str):
    print(f"✅ Added {peer_email} to {platform_str}")


def print_peer_added(peer: Peer):
    print(
        f"✅ Peer {peer.email} added successfully on {len(peer.platforms)} transport(s)"
    )
    for platform in peer.platforms:
        print(f"• {platform.module_path}")


def print_peer_request_sending(peer_email: str) -> None:
    print(f"🤝 Sending peer request to {peer_email}...")


def print_peer_request_resending(peer_email: str) -> None:
    print(f"♻️  Resending peer request to {peer_email}...")


def print_peer_request_accepting(peer_email: str) -> None:
    print(f"🤝 Accepting peer request from {peer_email}...")


def print_peer_request_sent(peer_email: str) -> None:
    print(f"\n✅ Peer request sent to {peer_email}!")
    print("   ⏳ Next step: ask them to approve your request.")
    print("   Once approved, run client.sync() to confirm the connection.")


def print_peer_connection_established(peer_email: str) -> None:
    print(f"\n✅ Connection with {peer_email} established!")
    print("   Run client.sync() to start syncing.")


def print_peer_already_connected(peer_email: str, state: str) -> None:
    print(f"ℹ️  Already have a connection with {peer_email} (state: {state}).")
    print("   Use force=True to resend the request.")
