import pytest
import yaml

from syft_perms import SyftPermContext, SyftFile, SyftFolder


OWNER = "alice@example.com"
USER = "bob@example.com"
USER2 = "charlie@example.com"


@pytest.fixture
def datasite(tmp_path):
    """Create a datasite directory with the owner email as folder name."""
    ds = tmp_path / OWNER
    ds.mkdir()
    return ds


@pytest.fixture
def sp(datasite):
    """Create a SyftPermContext instance."""
    return SyftPermContext(datasite=datasite)


# --- Step 1: Init ---


def test_init_extracts_owner(sp):
    assert sp.owner == OWNER


def test_init_loads_existing_yaml(datasite):
    # Create a yaml before init
    yaml_data = {
        "rules": [
            {"pattern": "**", "access": {"read": ["*"], "write": [], "admin": []}}
        ],
        "terminal": False,
    }
    yaml_path = datasite / "syft.pub.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump(yaml_data, f)

    sp = SyftPermContext(datasite=datasite)
    file = sp.open("test.txt")
    assert file.has_read_access(USER)


def test_open_file(sp, datasite):
    (datasite / "data.csv").touch()
    f = sp.open("data.csv")
    assert isinstance(f, SyftFile)


def test_open_folder_trailing_slash(sp, datasite):
    (datasite / "my_project").mkdir()
    f = sp.open("my_project/")
    assert isinstance(f, SyftFolder)


def test_open_folder_existing_dir(sp, datasite):
    (datasite / "my_project").mkdir()
    f = sp.open("my_project")
    assert isinstance(f, SyftFolder)


def test_open_nonexistent_returns_file(sp):
    f = sp.open("future.txt")
    assert isinstance(f, SyftFile)


# --- Step 2: Grant ---


def test_grant_read_access(sp):
    file = sp.open("data.csv")
    file.grant_read_access(USER)
    assert file.has_read_access(USER)


def test_grant_write_access(sp):
    file = sp.open("data.csv")
    file.grant_write_access(USER)
    assert file.has_write_access(USER)
    # Write implies read
    assert file.has_read_access(USER)


def test_grant_admin_access(sp):
    file = sp.open("data.csv")
    file.grant_admin_access(USER)
    assert file.has_admin_access(USER)
    assert file.has_write_access(USER)
    assert file.has_read_access(USER)


def test_grant_creates_yaml(sp, datasite):
    file = sp.open("data.csv")
    file.grant_read_access(USER)
    assert (datasite / "syft.pub.yaml").exists()


def test_grant_to_multiple_users(sp):
    file = sp.open("data.csv")
    file.grant_read_access(USER)
    file.grant_read_access(USER2)
    assert file.has_read_access(USER)
    assert file.has_read_access(USER2)


def test_grant_idempotent(sp, datasite):
    file = sp.open("data.csv")
    file.grant_read_access(USER)
    file.grant_read_access(USER)
    # Should not duplicate the user
    with open(datasite / "syft.pub.yaml") as f:
        data = yaml.safe_load(f)
    assert data["rules"][0]["access"]["read"].count(USER) == 1


# --- Step 3: Revoke ---


def test_revoke_read_access(sp):
    file = sp.open("data.csv")
    file.grant_read_access(USER)
    assert file.has_read_access(USER)
    file.revoke_read_access(USER)
    assert not file.has_read_access(USER)


def test_revoke_write_access(sp):
    file = sp.open("data.csv")
    file.grant_write_access(USER)
    file.revoke_write_access(USER)
    assert not file.has_write_access(USER)


def test_revoke_nonexistent_user_is_noop(sp):
    file = sp.open("data.csv")
    file.grant_read_access(USER)
    file.revoke_read_access(USER2)  # USER2 was never granted
    assert file.has_read_access(USER)


# --- Step 4: Check ---


def test_owner_always_has_access(sp):
    file = sp.open("data.csv")
    assert file.has_read_access(OWNER)
    assert file.has_write_access(OWNER)
    assert file.has_admin_access(OWNER)


def test_no_permission_denies(sp):
    file = sp.open("data.csv")
    assert not file.has_read_access(USER)
    assert not file.has_write_access(USER)
    assert not file.has_admin_access(USER)


def test_wildcard_grant(sp, datasite):
    yaml_data = {
        "rules": [
            {"pattern": "**", "access": {"read": ["*"], "write": [], "admin": []}}
        ],
        "terminal": False,
    }
    with open(datasite / "syft.pub.yaml", "w") as f:
        yaml.safe_dump(yaml_data, f)

    sp_reloaded = SyftPermContext(datasite=datasite)
    file = sp_reloaded.open("any_file.txt")
    assert file.has_read_access(USER)
    assert not file.has_write_access(USER)


# --- Step 5: Explain ---


def test_explain_owner(sp):
    file = sp.open("data.csv")
    expl = file.explain_permissions(OWNER)
    assert expl.is_owner
    assert expl.read
    assert expl.write
    assert expl.admin
    assert "Owner" in expl.reasons["read"]


def test_explain_no_permission(sp):
    file = sp.open("data.csv")
    expl = file.explain_permissions(USER)
    assert not expl.is_owner
    assert not expl.read
    assert "No permission file found" in expl.reasons["read"]


def test_explain_with_grant(sp):
    file = sp.open("data.csv")
    file.grant_read_access(USER)
    expl = file.explain_permissions(USER)
    assert expl.read
    assert not expl.write
    assert expl.governing_yaml is not None
    assert expl.matched_rule == "data.csv"


def test_explain_str(sp):
    file = sp.open("data.csv")
    file.grant_read_access(USER)
    expl = file.explain_permissions(USER)
    text = str(expl)
    assert "data.csv" in text
    assert USER in text


# --- Step 6: Files API ---


def test_files_browser_all(sp, datasite):
    (datasite / "a.txt").touch()
    (datasite / "b.txt").touch()
    (datasite / "subdir").mkdir()
    items = sp.files.all()
    assert len(items) == 2
    assert all(isinstance(i, SyftFile) for i in items)


def test_folders_browser_all(sp, datasite):
    (datasite / "subdir1").mkdir()
    (datasite / "subdir2").mkdir()
    items = sp.folders.all()
    assert len(items) == 2
    assert all(isinstance(i, SyftFolder) for i in items)


def test_files_and_folders_browser(sp, datasite):
    (datasite / "a.txt").touch()
    (datasite / "subdir").mkdir()
    items = sp.files_and_folders.all()
    assert len(items) == 2


def test_files_browser_get(sp, datasite):
    for i in range(5):
        (datasite / f"file{i}.txt").touch()
    items = sp.files.get(limit=2, offset=1)
    assert len(items) == 2


def test_files_browser_slice(sp, datasite):
    for i in range(5):
        (datasite / f"file{i}.txt").touch()
    items = sp.files[1:3]
    assert len(items) == 2


def test_files_browser_search(sp, datasite):
    (datasite / "a.txt").touch()
    (datasite / "b.txt").touch()
    file_a = sp.open("a.txt")
    file_a.grant_read_access(USER)
    results = sp.files.search(read=USER)
    assert len(results) == 1
    assert results[0].abs_path.name == "a.txt"


def test_files_browser_excludes_yaml(sp, datasite):
    (datasite / "a.txt").touch()
    file_a = sp.open("a.txt")
    file_a.grant_read_access(USER)  # This creates syft.pub.yaml
    items = sp.files.all()
    names = [i.abs_path.name for i in items]
    assert "syft.pub.yaml" not in names


# --- Step 7: Move ---


def test_move_file(sp, datasite):
    (datasite / "old.txt").write_text("hello")
    file = sp.open("old.txt")
    file.grant_read_access(USER)
    assert file.has_read_access(USER)

    new_file = file.move_file_and_its_permissions("new.txt")
    assert (datasite / "new.txt").exists()
    assert not (datasite / "old.txt").exists()
    assert new_file.has_read_access(USER)


def test_move_file_to_subdirectory(sp, datasite):
    (datasite / "old.txt").write_text("hello")
    file = sp.open("old.txt")
    file.grant_read_access(USER)

    new_file = file.move_file_and_its_permissions("subdir/moved.txt")
    assert (datasite / "subdir" / "moved.txt").exists()
    assert new_file.has_read_access(USER)


# --- Folder-specific ---


def test_folder_grant_read(sp, datasite):
    (datasite / "project").mkdir()
    folder = sp.open("project/")
    folder.grant_read_access(USER)
    # Files inside the folder should now be readable
    (datasite / "project" / "file.txt").touch()
    inner = sp.open("project/file.txt")
    assert inner.has_read_access(USER)


def test_folder_revoke(sp, datasite):
    (datasite / "project").mkdir()
    folder = sp.open("project/")
    folder.grant_read_access(USER)
    folder.revoke_read_access(USER)
    (datasite / "project" / "file.txt").touch()
    inner = sp.open("project/file.txt")
    assert not inner.has_read_access(USER)


def test_folder_has_access(sp, datasite):
    (datasite / "project").mkdir()
    folder = sp.open("project/")
    folder.grant_write_access(USER)
    assert folder.has_write_access(USER)
    assert folder.has_read_access(USER)
    assert not folder.has_admin_access(USER)


def test_folder_explain(sp, datasite):
    (datasite / "project").mkdir()
    folder = sp.open("project/")
    folder.grant_read_access(USER)
    expl = folder.explain_permissions(USER)
    assert expl.read


# --- Nested permissions ---


def test_nested_permission_override(sp, datasite):
    # Root grants read to everyone
    root_yaml = {
        "rules": [
            {"pattern": "**", "access": {"read": ["*"], "write": [], "admin": []}}
        ],
        "terminal": False,
    }
    with open(datasite / "syft.pub.yaml", "w") as f:
        yaml.safe_dump(root_yaml, f)

    # Subfolder grants write to specific user
    sub = datasite / "sub"
    sub.mkdir()
    sub_yaml = {
        "rules": [
            {"pattern": "**", "access": {"read": [], "write": [USER], "admin": []}}
        ],
        "terminal": False,
    }
    with open(sub / "syft.pub.yaml", "w") as f:
        yaml.safe_dump(sub_yaml, f)

    sp_new = SyftPermContext(datasite=datasite)
    # File in root: readable by everyone
    root_file = sp_new.open("test.txt")
    assert root_file.has_read_access(USER2)

    # File in sub: governed by sub's yaml (closest wins, no merge)
    sub_file = sp_new.open("sub/test.txt")
    assert sub_file.has_write_access(USER)
    # USER2 has no access via sub yaml
    assert not sub_file.has_read_access(USER2)


# --- Resolve correct yaml ---


def test_grant_resolves_to_governing_terminal_yaml(datasite):
    """When a terminal yaml governs a nested file, grant should add rule there."""
    sub = datasite / "sub"
    sub.mkdir()

    yaml_data = {
        "rules": [{"pattern": "**", "access": {"read": [], "write": [], "admin": []}}],
        "terminal": True,
    }
    with open(datasite / "syft.pub.yaml", "w") as f:
        yaml.safe_dump(yaml_data, f)

    sp = SyftPermContext(datasite=datasite)
    file = sp.open("sub/data.csv")
    file.grant_read_access(USER)

    # Should NOT create sub/syft.pub.yaml
    assert not (sub / "syft.pub.yaml").exists()
    # Should add specific rule to root yaml
    with open(datasite / "syft.pub.yaml") as f:
        data = yaml.safe_load(f)
    patterns = [r["pattern"] for r in data["rules"]]
    assert "sub/data.csv" in patterns
    # Verify access works
    assert file.has_read_access(USER)


def test_folder_grant_resolves_to_governing_terminal_yaml(datasite):
    """a/b/syft.pub.yaml terminal -> add c/** rule to a/b/syft.pub.yaml."""
    ab = datasite / "a" / "b"
    ab.mkdir(parents=True)
    abc = ab / "c"
    abc.mkdir()

    yaml_data = {
        "rules": [{"pattern": "**", "access": {"read": [], "write": [], "admin": []}}],
        "terminal": True,
    }
    with open(ab / "syft.pub.yaml", "w") as f:
        yaml.safe_dump(yaml_data, f)

    sp = SyftPermContext(datasite=datasite)
    folder = sp.open("a/b/c/")
    folder.grant_read_access(USER)

    # Should NOT create a/b/c/syft.pub.yaml
    assert not (abc / "syft.pub.yaml").exists()
    # Should add c/** rule to a/b/syft.pub.yaml
    with open(ab / "syft.pub.yaml") as f:
        data = yaml.safe_load(f)
    patterns = [r["pattern"] for r in data["rules"]]
    assert "c/**" in patterns
    # Verify access works for files inside the folder
    (abc / "file.txt").touch()
    inner = sp.open("a/b/c/file.txt")
    assert inner.has_read_access(USER)


def test_grant_modifies_exact_rule_in_governing_yaml(datasite):
    """When governing yaml has exact rule for a file, modify that rule."""
    yaml_data = {
        "rules": [
            {
                "pattern": "data.csv",
                "access": {"read": [USER], "write": [], "admin": []},
            }
        ],
        "terminal": False,
    }
    with open(datasite / "syft.pub.yaml", "w") as f:
        yaml.safe_dump(yaml_data, f)

    sp = SyftPermContext(datasite=datasite)
    file = sp.open("data.csv")
    file.grant_write_access(USER)

    with open(datasite / "syft.pub.yaml") as f:
        data = yaml.safe_load(f)
    # Should have modified the existing rule, not created a new one
    assert len(data["rules"]) == 1
    assert USER in data["rules"][0]["access"]["write"]
    assert USER in data["rules"][0]["access"]["read"]


def test_revoke_resolves_to_governing_terminal_yaml(datasite):
    """Revoke should resolve to the governing yaml with the correct pattern."""
    sub = datasite / "sub"
    sub.mkdir()

    yaml_data = {
        "rules": [
            {
                "pattern": "sub/data.csv",
                "access": {"read": [USER], "write": [], "admin": []},
            }
        ],
        "terminal": True,
    }
    with open(datasite / "syft.pub.yaml", "w") as f:
        yaml.safe_dump(yaml_data, f)

    sp = SyftPermContext(datasite=datasite)
    file = sp.open("sub/data.csv")
    assert file.has_read_access(USER)

    file.revoke_read_access(USER)
    assert not file.has_read_access(USER)


def test_folder_revoke_resolves_to_governing_yaml(datasite):
    """Folder revoke should resolve to the governing yaml."""
    ab = datasite / "a" / "b"
    ab.mkdir(parents=True)
    abc = ab / "c"
    abc.mkdir()

    yaml_data = {
        "rules": [
            {"pattern": "c/**", "access": {"read": [USER], "write": [], "admin": []}}
        ],
        "terminal": True,
    }
    with open(ab / "syft.pub.yaml", "w") as f:
        yaml.safe_dump(yaml_data, f)

    sp = SyftPermContext(datasite=datasite)
    folder = sp.open("a/b/c/")
    assert folder.has_read_access(USER)

    folder.revoke_read_access(USER)
    assert not folder.has_read_access(USER)


# --- Revoke wildcard entries ---


def test_revoke_removes_wildcard_star(datasite):
    """Revoking a user who has access via '*' removes the '*' entry with a warning."""
    yaml_data = {
        "rules": [
            {"pattern": "**", "access": {"read": ["*"], "write": [], "admin": []}}
        ],
        "terminal": False,
    }
    with open(datasite / "syft.pub.yaml", "w") as f:
        yaml.safe_dump(yaml_data, f)

    sp = SyftPermContext(datasite=datasite)
    file = sp.open("data.csv")
    assert file.has_read_access(USER)

    with pytest.warns(UserWarning, match="Removed '\\*'.*may affect other users"):
        file.revoke_read_access(USER)
    assert not file.has_read_access(USER)


def test_revoke_removes_domain_wildcard(datasite):
    """Revoking a user matched by '*@example.com' removes that entry with a warning."""
    yaml_data = {
        "rules": [
            {
                "pattern": "**",
                "access": {"read": ["*@example.com"], "write": [], "admin": []},
            }
        ],
        "terminal": False,
    }
    with open(datasite / "syft.pub.yaml", "w") as f:
        yaml.safe_dump(yaml_data, f)

    sp = SyftPermContext(datasite=datasite)
    file = sp.open("data.csv")
    assert file.has_read_access(USER)

    with pytest.warns(
        UserWarning, match="Removed '\\*@example.com'.*may affect other users"
    ):
        file.revoke_read_access(USER)
    assert not file.has_read_access(USER)


def test_revoke_removes_user_placeholder(datasite):
    """Revoking a user matched by 'USER' removes that entry with a warning."""
    yaml_data = {
        "rules": [
            {
                "pattern": "{{.UserEmail}}/**",
                "access": {"read": ["USER"], "write": [], "admin": []},
            }
        ],
        "terminal": False,
    }
    with open(datasite / "syft.pub.yaml", "w") as f:
        yaml.safe_dump(yaml_data, f)

    sp = SyftPermContext(datasite=datasite)
    file = sp.open(f"{USER}/data.csv")
    assert file.has_read_access(USER)

    with pytest.warns(UserWarning, match="Removed 'USER'.*may affect other users"):
        file.revoke_read_access(USER)
    assert not file.has_read_access(USER)


# --- syft.pub.yaml requires admin ---


def test_syft_pub_yaml_requires_admin_to_write(datasite):
    """Accessing syft.pub.yaml itself requires ADMIN level.

    A user with only write access should NOT be able to write/modify syft.pub.yaml.
    A user with admin access CAN write/modify syft.pub.yaml.
    """
    yaml_data = {
        "rules": [
            {
                "pattern": "**",
                "access": {
                    "read": [USER],
                    "write": [USER],
                    "admin": [USER2],
                },
            }
        ],
        "terminal": False,
    }
    perm_dir = datasite / "project"
    perm_dir.mkdir()
    with open(perm_dir / "syft.pub.yaml", "w") as f:
        yaml.safe_dump(yaml_data, f)

    sp = SyftPermContext(datasite=datasite)

    perm_file = sp.open("project/syft.pub.yaml")

    # USER has write but NOT admin → cannot write syft.pub.yaml
    assert not perm_file.has_write_access(USER)

    # USER2 has admin → can write syft.pub.yaml
    assert perm_file.has_write_access(USER2)
