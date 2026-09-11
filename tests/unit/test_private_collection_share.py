"""Private datasets travel to an enclave as streamed, encrypted collections.

Covers the three properties the design promises:
- memory: sharing and receiving a file never holds the file in memory,
- dedup: a repeat share for the same audience uploads nothing, a new audience
  gets its own collection,
- reach: the enclave pulls the shared collection into the path jobs resolve,
  while a peer that was not shared on it sees nothing.
"""

import os
import random
import resource
import sys
from pathlib import Path

import pytest
from syft.sync.peers.peer import Peer
from syft.sync.peers.peer_store import PeerStore
from syft_enclaves import SyftEnclaveClient
from syft_rds.config import PRIVATE_DATASET_COLLECTION_PREFIX

MiB = 1024 * 1024


def _peak_rss() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024


def _write_random(path: Path, size: int) -> None:
    with open(path, "wb") as fh:
        remaining = size
        while remaining > 0:
            chunk = min(remaining, MiB)
            fh.write(os.urandom(chunk))
            remaining -= chunk


def _same_content(a: Path, b: Path) -> bool:
    if a.stat().st_size != b.stat().st_size:
        return False
    with open(a, "rb") as fa, open(b, "rb") as fb:
        while True:
            x, y = fa.read(MiB), fb.read(MiB)
            if x != y:
                return False
            if not x:
                return True


# ============================================================================
# peer_store: streaming file encryption
# ============================================================================


@pytest.fixture
def stores():
    alice = PeerStore(email="alice@example.com", use_encryption=True)
    alice.generate_keys()
    bob = PeerStore(email="bob@example.com", use_encryption=True)
    bob.generate_keys()
    alice.add_peer(Peer(email="bob@example.com", use_encryption=True))
    bob.add_peer(Peer(email="alice@example.com", use_encryption=True))
    alice.set_peer_bundle("bob@example.com", bob.get_public_bundle())
    bob.set_peer_bundle("alice@example.com", alice.get_public_bundle())
    return alice, bob


def _round_trip_growth(alice, bob, tmp_path: Path, size: int) -> int:
    """Peak-RSS growth of encrypting a `size` file for peer+self and opening it both ways."""
    src = tmp_path / f"weights-{size}.bin"
    _write_random(src, size)
    before = _peak_rss()

    enc = tmp_path / f"weights-{size}.syc"
    alice.encrypt_file_for(["bob@example.com", "alice@example.com"], src, enc)
    assert PeerStore.is_envelope_file(enc)
    assert not PeerStore.is_envelope_file(src)

    # Bob opens it as a peer; Alice opens the same file as self (her backup).
    out_bob = tmp_path / f"bob-{size}.out"
    bob.decrypt_collection_file("alice@example.com", enc, out_bob)
    assert _same_content(src, out_bob)
    assert not enc.exists(), "decrypt_collection_file consumes its input"

    alice.encrypt_file_for(["bob@example.com", "alice@example.com"], src, enc)
    out_alice = tmp_path / f"alice-{size}.out"
    alice.decrypt_collection_file("alice@example.com", enc, out_alice)
    assert _same_content(src, out_alice)
    return _peak_rss() - before


def test_file_round_trip_for_peer_and_self_is_memory_bounded(tmp_path, stores):
    alice, bob = stores
    # Memory must not grow with the file. Working-set noise between runs is on
    # the order of the parallel segment buffers (tens of MiB), so the bound is
    # "far below the file size", not "zero": holding a 96 MiB file even once
    # would blow well past half of it.
    small = _round_trip_growth(alice, bob, tmp_path, 32 * MiB)
    large = _round_trip_growth(alice, bob, tmp_path, 96 * MiB)
    assert large < 48 * MiB, (
        f"a 96 MiB round trip grew RSS by {large / MiB:.1f} MiB: the file is being buffered"
    )
    assert small < 96 * MiB, f"first round trip grew RSS by {small / MiB:.1f} MiB"


def test_file_decrypt_rejects_wrong_sender_and_tampering(tmp_path, stores):
    alice, bob = stores
    carol = PeerStore(email="carol@example.com", use_encryption=True)
    carol.generate_keys()
    bob.add_peer(Peer(email="carol@example.com", use_encryption=True))
    bob.set_peer_bundle("carol@example.com", carol.get_public_bundle())

    src = tmp_path / "data.bin"
    _write_random(src, 2 * MiB + 17)
    enc = tmp_path / "data.syc"
    alice.encrypt_file_for(["bob@example.com"], src, enc)

    # Claimed sender does not match the signature.
    with pytest.raises(Exception):
        bob.decrypt_file("carol@example.com", enc, tmp_path / "x.out")
    assert not (tmp_path / "x.out").exists()

    # A flipped byte in the payload.
    data = bytearray(enc.read_bytes())
    data[-40] ^= 0x01
    enc.write_bytes(bytes(data))
    with pytest.raises(Exception):
        bob.decrypt_file("alice@example.com", enc, tmp_path / "y.out")
    assert not (tmp_path / "y.out").exists()


def test_plaintext_collection_file_passes_through(tmp_path, stores):
    alice, bob = stores
    src = tmp_path / "mock.csv"
    src.write_bytes(b"a,b\n1,2\n")
    dest = tmp_path / "dest" / "mock.csv"
    bob.decrypt_collection_file("alice@example.com", src, dest)
    assert dest.read_bytes() == b"a,b\n1,2\n"
    assert not src.exists()


# ============================================================================
# end to end: owner -> enclave through the in-memory quad
# ============================================================================


def _private_collections_of(client, tag: str) -> list:
    router = client._rds.sync_engine._connection_router
    return [
        c
        for c in router.owner_list_all_collections_with_permissions(
            PRIVATE_DATASET_COLLECTION_PREFIX
        )
        if c.tag == tag
    ]


def _private_dir_on(client, owner_email: str, tag: str) -> Path | None:
    root = client._rds.syftbox_folder / owner_email / "private" / "syft_datasets"
    if not root.exists():
        return None
    hits = [p for p in root.rglob(tag) if p.is_dir()]
    assert len(hits) <= 1, hits
    return hits[0] if hits else None


def test_private_share_streams_dedups_and_reaches_only_the_enclave(tmp_path):
    enclave, do1, do2, ds = SyftEnclaveClient.quad_with_mock_drive_service_connection(
        use_in_memory_cache=False,
        encryption=True,
    )

    src_dir = tmp_path / str(random.randint(1, 10**6))
    src_dir.mkdir()
    mock_path = src_dir / "mock.txt"
    mock_path.write_text("mock")
    private_dir = src_dir / "private"
    (private_dir / "ckpt").mkdir(parents=True)
    private_path = private_dir / "model.bin"
    _write_random(private_path, 32 * MiB)
    # A nested file, as an Orbax checkpoint directory would be laid out.
    nested = private_dir / "ckpt" / "shard-0"
    nested.write_bytes(b"nested shard")

    do1.create_dataset(
        name="bigmodel",
        mock_path=mock_path,
        private_path=private_dir,
        summary="big private file",
        users=[ds.email],
        upload_private=True,
        sync=False,
    )
    # One private collection: the owner's own backup, sealed for self.
    backup = _private_collections_of(do1, "bigmodel")
    assert len(backup) == 1

    # No RSS assertion here on purpose: the mock Drive service keeps every
    # uploaded file in memory, so process growth measures the mock, not the
    # client. Memory bounds are asserted on the peer_store path above and in
    # the crypto library's own tests.
    do1.share_private_dataset("bigmodel", enclave.email)
    do1.sync()
    enclave._rds.sync()

    # The enclave holds the file where jobs resolve it, byte for byte.
    enclave_dir = _private_dir_on(enclave, do1.email, "bigmodel")
    assert enclave_dir is not None
    received = enclave_dir / "model.bin"
    assert received.exists() and _same_content(private_path, received)
    assert not PeerStore.is_envelope_file(received), "must be decrypted on arrival"
    # Nested files keep their relative path on the receiver.
    assert (enclave_dir / "ckpt" / "shard-0").read_bytes() == b"nested shard"

    # A second collection exists: same files, audience {owner, enclave}.
    shared = _private_collections_of(do1, "bigmodel")
    assert len(shared) == 2
    assert backup[0].content_hash in {c.content_hash for c in shared}

    # Repeat share for the same audience: nothing new is created.
    do1.share_private_dataset("bigmodel", enclave.email)
    assert len(_private_collections_of(do1, "bigmodel")) == 2

    # A new peer is a new audience: one more collection, the others untouched.
    do1.share_private_dataset("bigmodel", do2.email)
    assert len(_private_collections_of(do1, "bigmodel")) == 3

    # The data scientist was never shared on the private data and sees none of it.
    ds._rds.sync()
    assert _private_dir_on(ds, do1.email, "bigmodel") is None

    # Syncing the enclave again downloads nothing: the hash cache skips it.
    mtime = received.stat().st_mtime_ns
    enclave._rds.sync()
    assert received.stat().st_mtime_ns == mtime
