"""``upgrade()`` promotes owned datasets to the layout of the installed client.

A dataset's object versions follow its layout directory, so a promote is the
addition of the current layout. These tests drive the sweep through the mock
Drive: what it publishes, what it skips, who it reaches, and what it reports.
"""

import pytest
from syft_datasets.dataset_manager import SHARE_WITH_ANY
from syft_datasets.migrations.registry import DATASET_PROTOCOL_VERSION
from syft_rds import SyftRDSClient
from syft_rds.config import MOCK_DATASET_SPEC, dataset_variant

from dataset_test_utils import create_tmp_dataset_files

OLD_PEER = "old@test.org"


@pytest.fixture
def pair():
    return SyftRDSClient.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False,
    )


def _published(do_manager, tag: str) -> set:
    return {do_manager._protocol_of(c) for c in do_manager._mock_collections_for(tag)}


def _create_v0_dataset(do_manager, tag: str, users=None, upload_private: bool = False):
    """A dataset of an earlier release: the flat layout, published and shared."""
    mock_path, private_path, readme_path = create_tmp_dataset_files()
    created = do_manager.dataset_manager.create_all(
        name=tag,
        mock_path=mock_path,
        private_path=private_path,
        readme_path=readme_path,
        users=users,
        protocol_versions=["0"],
    )
    do_manager._upload_dataset_to_collection(created["0"], users=users or [])
    if upload_private:
        do_manager._upload_private_dataset_to_collection(created["0"])
    return created["0"]


def test_upgrade_promotes_a_v0_dataset_and_publishes_it(pair):
    ds_manager, do_manager = pair
    _create_v0_dataset(do_manager, "legacy", users=[ds_manager.email])
    assert _published(do_manager, "legacy") == {"0"}

    report = do_manager.upgrade(sync=False)

    assert _published(do_manager, "legacy") == {"0", DATASET_PROTOCOL_VERSION}
    assert [d.tag for d in report.promoted] == ["legacy"]
    assert report.datasets[0].published_before == ["0"]
    assert report.datasets[0].published_after == ["0", DATASET_PROTOCOL_VERSION]
    assert report.upload_bytes > 0
    assert not report.failed


def test_the_promoted_copy_reaches_the_audience(pair):
    ds_manager, do_manager = pair
    _create_v0_dataset(do_manager, "legacy", users=[ds_manager.email])

    do_manager.upgrade(sync=False)
    ds_manager.sync()

    dataset = ds_manager.datasets.get("legacy", datasite=do_manager.email)
    # The DS reads the newest layout it can, which is the promoted copy.
    assert dataset.protocol_version == DATASET_PROTOCOL_VERSION
    assert dataset.mock_files
    for path in dataset.mock_files:
        assert path.exists()


def test_upgrade_publishes_a_layout_that_is_on_disk_but_unpublished(pair):
    # The regression test for the published-only skip: a migrate can succeed and
    # the upload fail, so the layout exists locally with no collection. A skip
    # that folded the disk into the published set would never publish it.
    ds_manager, do_manager = pair
    _create_v0_dataset(do_manager, "halfway", users=[ds_manager.email])
    do_manager.dataset_manager.migrate(
        "halfway", DATASET_PROTOCOL_VERSION, users=[ds_manager.email]
    )
    storage = do_manager.dataset_manager.storage
    assert storage.find_dataset_ref(
        do_manager.email, "halfway", protocol_version=DATASET_PROTOCOL_VERSION
    )
    assert _published(do_manager, "halfway") == {"0"}

    report = do_manager.upgrade(sync=False)

    assert _published(do_manager, "halfway") == {"0", DATASET_PROTOCOL_VERSION}
    assert [d.tag for d in report.promoted] == ["halfway"]


def test_upgrade_is_a_no_op_when_the_current_layout_is_published(pair):
    ds_manager, do_manager = pair
    mock_path, private_path, readme_path = create_tmp_dataset_files()
    do_manager.create_dataset(
        name="current",
        mock_path=mock_path,
        private_path=private_path,
        readme_path=readme_path,
        users=[ds_manager.email],
        sync=False,
    )
    before = _published(do_manager, "current")

    report = do_manager.upgrade(sync=False)

    assert _published(do_manager, "current") == before
    assert not report.promoted
    assert "already published" in report.datasets[0].skipped_reason


def test_dry_run_writes_nothing_and_reports_the_upload_cost(pair):
    ds_manager, do_manager = pair
    _create_v0_dataset(do_manager, "legacy", users=[ds_manager.email])
    storage = do_manager.dataset_manager.storage
    target = storage.new_dataset_ref("legacy", DATASET_PROTOCOL_VERSION)

    report = do_manager.upgrade(dry_run=True)

    assert report.dry_run
    assert _published(do_manager, "legacy") == {"0"}
    assert not storage.public_dataset_dir(target).exists()
    assert report.datasets[0].published_after == ["0", DATASET_PROTOCOL_VERSION]
    assert report.upload_bytes > 0


def test_dry_run_applies_the_same_skip_as_the_sweep(pair):
    ds_manager, do_manager = pair
    mock_path, private_path, readme_path = create_tmp_dataset_files()
    do_manager.create_dataset(
        name="current",
        mock_path=mock_path,
        private_path=private_path,
        readme_path=readme_path,
        users=[ds_manager.email],
        sync=False,
    )

    report = do_manager.upgrade(dry_run=True)

    assert not report.promoted
    assert report.upload_bytes == 0


def test_an_owner_only_dataset_stays_owner_only(pair):
    _, do_manager = pair
    _create_v0_dataset(do_manager, "mine")

    do_manager.upgrade(sync=False)

    promoted = [
        c
        for c in do_manager._mock_collections_for("mine")
        if do_manager._protocol_of(c) == DATASET_PROTOCOL_VERSION
    ]
    assert len(promoted) == 1
    assert not promoted[0].has_any_permission
    # No ruleset is written for an audience of nobody.
    storage = do_manager.dataset_manager.storage
    target = storage.new_dataset_ref("mine", DATASET_PROTOCOL_VERSION)
    assert not (storage.public_dataset_dir(target) / "syft.pub.yaml").exists()


def test_an_any_dataset_is_still_any_after_the_upgrade(pair):
    # The "any" flag lives on the collection, not in the ruleset, and it is the
    # wider audience, so it has priority over the recovered emails.
    _, do_manager = pair
    _create_v0_dataset(do_manager, "open", users=SHARE_WITH_ANY)
    assert any(c.has_any_permission for c in do_manager._mock_collections_for("open"))

    do_manager.upgrade(sync=False)

    promoted = [
        c
        for c in do_manager._mock_collections_for("open")
        if do_manager._protocol_of(c) == DATASET_PROTOCOL_VERSION
    ]
    assert len(promoted) == 1
    assert promoted[0].has_any_permission


def test_a_user_added_after_create_is_on_the_promoted_copy(pair):
    # The drift case. A share updates the transport, so the ruleset of every
    # layout must record it too, or the recovered audience is the create-time
    # one and the promoted copy never reaches the later user.
    ds_manager, do_manager = pair
    _create_v0_dataset(do_manager, "shared", users=[ds_manager.email])
    do_manager.share_dataset("shared", [OLD_PEER], sync=False)

    do_manager.upgrade(sync=False)

    audience = do_manager.dataset_manager.recover_audience_from_ruleset(
        "shared", protocol_version=DATASET_PROTOCOL_VERSION
    )
    assert sorted(audience) == sorted([ds_manager.email, OLD_PEER])


def test_one_failing_dataset_is_reported_and_the_rest_continue(pair):
    ds_manager, do_manager = pair
    _create_v0_dataset(do_manager, "aaa broken", users=[ds_manager.email])
    _create_v0_dataset(do_manager, "zzz fine", users=[ds_manager.email])

    real_materialize = do_manager._materialize_dataset_copy

    def fail_for_one(tag, protocol_version, users):
        if tag == "aaa broken":
            raise RuntimeError("migrate exploded")
        return real_materialize(tag, protocol_version, users)

    do_manager._materialize_dataset_copy = fail_for_one
    try:
        report = do_manager.upgrade(sync=False)
    finally:
        do_manager._materialize_dataset_copy = real_materialize

    failed = {d.tag for d in report.failed}
    promoted = {d.tag for d in report.promoted}
    assert failed == {"aaa broken"}
    assert promoted == {"zzz fine"}
    assert "migrate exploded" in report.datasets[0].error
    assert _published(do_manager, "zzz fine") == {"0", DATASET_PROTOCOL_VERSION}


def test_upgrade_is_refused_without_the_do_role(pair):
    ds_manager, _ = pair
    with pytest.raises(ValueError, match="Only dataset owners"):
        ds_manager.upgrade()


def test_a_share_that_fails_after_the_upload_is_retried_next_sweep(pair):
    # The upload and the share are separate steps. If the share fails, the
    # layout is published but unshared, and a skip on "already published" would
    # strand the audience on the old copy with no way back.
    ds_manager, do_manager = pair
    _create_v0_dataset(do_manager, "flaky", users=[ds_manager.email])

    real_share = do_manager._share_dataset_collection
    calls = []

    def fail_once(wire_prefix, tag, content_hash, users):
        calls.append((tag, content_hash, users))
        raise RuntimeError("share exploded")

    do_manager._share_dataset_collection = fail_once
    try:
        first = do_manager.upgrade(sync=False)
    finally:
        do_manager._share_dataset_collection = real_share

    # The copy went up, the share did not, and the failure is reported.
    assert _published(do_manager, "flaky") == {"0", DATASET_PROTOCOL_VERSION}
    assert "share exploded" in first.datasets[0].error
    assert not first.promoted

    shared = []

    def record(wire_prefix, tag, content_hash, users):
        shared.append((wire_prefix, users))
        return real_share(wire_prefix, tag, content_hash, users)

    do_manager._share_dataset_collection = record
    try:
        second = do_manager.upgrade(sync=False)
    finally:
        do_manager._share_dataset_collection = real_share

    # The second sweep re-shares every layout, including the promoted one.
    assert not second.failed
    assert {prefix for prefix, _ in shared} == {
        MOCK_DATASET_SPEC.wire_prefix(dataset_variant(v))
        for v in ("0", DATASET_PROTOCOL_VERSION)
    }
    assert all(users == [ds_manager.email] for _, users in shared)


def test_dry_run_counts_the_private_payload_of_a_drive_backed_dataset(pair):
    # A promote uploads a private collection too when the copies are
    # Drive-backed, so the reported cost must include it.
    ds_manager, do_manager = pair
    _create_v0_dataset(do_manager, "mock only", users=[ds_manager.email])
    _create_v0_dataset(
        do_manager, "with private", users=[ds_manager.email], upload_private=True
    )

    report = do_manager.upgrade(dry_run=True)
    by_tag = {d.tag: d.upload_bytes for d in report.datasets}

    assert by_tag["with private"] > by_tag["mock only"]
