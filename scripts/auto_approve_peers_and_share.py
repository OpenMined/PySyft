"""
Auto-approve peer requests from a specific email domain and share a dataset with them.

Environment variables:
    EMAIL: Data owner's email address
    TOKEN_PATH: Path to Google OAuth token file
    APPROVED_DOMAIN: Email domain to auto-approve (e.g., "openmined.org")
    DATASET_NAME: Name of the dataset to share with approved peers
"""

import os
import time
from pathlib import Path

from syft_client.sync.syftbox_manager import SyftboxManager
from syft_client.sync.peers.peer import PeerState


def get_email_domain(email: str) -> str:
    """Extract domain from email address."""
    return email.split("@")[-1].lower()


def main():
    email = os.environ.get("EMAIL")
    token_path = os.environ.get("TOKEN_PATH")
    approved_domain = os.environ.get("APPROVED_DOMAIN")
    dataset_name = os.environ.get("DATASET_NAME")

    if not email:
        raise ValueError("EMAIL environment variable is required")
    if not token_path:
        raise ValueError("TOKEN_PATH environment variable is required")
    if not approved_domain:
        raise ValueError("APPROVED_DOMAIN environment variable is required")
    if not dataset_name:
        raise ValueError("DATASET_NAME environment variable is required")

    approved_domain = approved_domain.lower()
    token_path = Path(token_path)

    if not token_path.exists():
        raise ValueError(f"Token file not found: {token_path}")

    print(f"Starting auto-approval service for domain: {approved_domain}")
    print(f"Dataset to share: {dataset_name}")
    print(f"Data owner email: {email}")

    client = SyftboxManager.for_jupyter(
        email=email,
        has_do_role=True,
        token_path=token_path,
    )

    # Track users we've already shared the dataset with to avoid duplicate API calls
    shared_with: set[str] = set()

    print("Polling for peer requests every 5 seconds...")

    while True:
        try:
            # Load latest peers
            client.load_peers()

            # Check for pending requests
            for peer in client.peers:
                if peer.state != PeerState.REQUESTED_BY_PEER:
                    continue

                peer_domain = get_email_domain(peer.email)
                if peer_domain != approved_domain:
                    print(
                        f"Skipping peer {peer.email} (domain {peer_domain} != {approved_domain})"
                    )
                    continue

                print(f"Found peer request from {peer.email} with matching domain")

                # Approve the peer request
                client.approve_peer_request(peer.email)

                # Share dataset if not already shared
                if peer.email not in shared_with:
                    print(f"Sharing dataset '{dataset_name}' with {peer.email}")
                    client.share_dataset(
                        tag=dataset_name, users=[peer.email], sync=True
                    )
                    shared_with.add(peer.email)
                else:
                    print(f"Dataset already shared with {peer.email}")

        except Exception as e:
            print(f"Error during polling: {e}")

        time.sleep(5)


if __name__ == "__main__":
    main()
