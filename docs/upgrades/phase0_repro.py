"""
Phase-0 reproduction: what actually happens to a user's Drive folders across a
version change in syft-client.

Exercises the REAL functions in gdrive_transport.py (no mocks of the logic
under test): the folder-name builders, `_filter_patch_compatible`, and
`_expect_one`. We only monkeypatch the module-level SYFT_CLIENT_VERSION to
play the role of "the client binary the user is currently running".

Run: python scratch_phase0_repro.py
"""

import syft_client.sync.connections.drive.gdrive_transport as gt

EMAIL = "alice@openmined.org"


def folder_names_at(version: str) -> dict[str, str]:
    """The folder names the WRITE path mints while running `version`."""
    gt.SYFT_CLIENT_VERSION = version
    return {
        "personal": gt.GdrivePersonalSyftboxFolder(email=EMAIL).as_string(),
        "p2p_outbox": gt.GdriveP2PFolder(
            datasite_email=EMAIL, folder_type="outbox", peer_email="bob@openmined.org"
        ).as_string(),
        "checkpoints": f"{EMAIL}-{gt.SYFT_CLIENT_VERSION}-checkpoints",
    }


def lookup(existing_on_drive: list[str], running_version: str):
    """Replay the READ path: filter Drive folders for the running client.

    Returns (resolved_folder_id_or_None, error_or_None).
    `_expect_one` is called unbound (it touches no instance state).
    """
    gt.SYFT_CLIENT_VERSION = running_version
    # (id, name) pairs as _find_folders would return them
    folders = [(f"id::{name}", name) for name in existing_on_drive]
    compatible = gt._filter_patch_compatible(folders)
    try:
        return gt.GDriveConnection._expect_one(None, compatible), None
    except RuntimeError as e:
        return None, str(e)


def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


banner("FACT 1 — folder names embed the FULL running version (write path)")
for v in ("0.1.111", "0.1.117", "0.2.0"):
    print(f"\n  running {v}:")
    for k, name in folder_names_at(v).items():
        print(f"    {k:12} -> {name}")

banner("SCENARIO A — solo PATCH upgrade, one old folder on Drive (the #280 fix)")
# User ran 0.1.111, created their folder, then upgraded the binary to 0.1.117.
drive = [gt.GdrivePersonalSyftboxFolder(email=EMAIL).as_string()  # minted under...
         for gt.SYFT_CLIENT_VERSION in ["0.1.111"]]
resolved, err = lookup(drive, running_version="0.1.117")
print(f"\n  on Drive : {drive}")
print(f"  running  : 0.1.117")
print(f"  resolved : {resolved}")
print(f"  error    : {err}")
print("  VERDICT  : " + ("FOUND — assets NOT stranded (patch-compat discovery works)"
                         if resolved else "STRANDED"))

banner("SCENARIO B — TWO patch folders coexist (pre-#280 residue / straddled upgrade)")
# e.g. a 0.1.111 folder AND a 0.1.112 folder both exist (0.1.112 was created
# before #280 taught discovery to look across patches).
gt.SYFT_CLIENT_VERSION = "0.1.111"
f_111 = gt.GdrivePersonalSyftboxFolder(email=EMAIL).as_string()
gt.SYFT_CLIENT_VERSION = "0.1.112"
f_112 = gt.GdrivePersonalSyftboxFolder(email=EMAIL).as_string()
drive = [f_111, f_112]
resolved, err = lookup(drive, running_version="0.1.117")
print(f"\n  on Drive : {drive}")
print(f"  running  : 0.1.117")
print(f"  resolved : {resolved}")
print(f"  error    : {err}")
print("  VERDICT  : " + ("HARD FAIL — RuntimeError, user must hand-delete a folder on Drive"
                         if err else "ok"))

banner("SCENARIO C — MINOR bump (the V1/V2 boundary): old folder goes invisible")
# User's data lives in a 0.1.x folder; they run a 0.2.0 binary.
gt.SYFT_CLIENT_VERSION = "0.1.117"
old = gt.GdrivePersonalSyftboxFolder(email=EMAIL).as_string()
resolved, err = lookup([old], running_version="0.2.0")
print(f"\n  on Drive : {[old]}")
print(f"  running  : 0.2.0")
print(f"  resolved : {resolved}")
print(f"  error    : {err}")
print("  VERDICT  : " + ("STRANDED — minor mismatch filtered out (forced-upgrade territory = V2)"
                         if not resolved and not err else "found"))
