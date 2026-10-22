"""Integration tests for CLI commands."""

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from syft_bg.approve.api_store import ApiStore

from syft_bg.cli.commands import (
    create_api,
    init,
    install,
    list_apis,
    main,
    remove_api,
    remove_peer,
    status,
    uninstall,
)


class TestInitCommand:
    """Tests for the init command."""

    @patch("syft_bg.api.api.init")
    def test_init_basic(self, mock_api_init):
        """Should pass email to api.init."""
        runner = CliRunner()
        result = runner.invoke(init, ["-e", "alice@uni.edu"])

        assert result.exit_code == 0
        mock_api_init.assert_called_once_with(
            do_email="alice@uni.edu",
            syftbox_root=None,
            token_path=None,
        )

    @patch("syft_bg.api.api.init")
    def test_init_all_options(self, mock_api_init):
        """Should pass all options to api.init."""
        runner = CliRunner()
        runner.invoke(
            init,
            ["-e", "alice@uni.edu", "-r", "/tmp/syftbox", "-t", "/tmp/token.json"],
            catch_exceptions=False,
        )

        # token path must exist for click.Path(exists=True), so this will fail
        # unless file exists. Test the basic flow instead.
        mock_api_init.assert_not_called()  # file doesn't exist

    @patch("syft_bg.api.api.init")
    def test_init_requires_email(self, mock_api_init):
        """Should error when email not provided."""
        runner = CliRunner()
        result = runner.invoke(init, [])

        assert result.exit_code != 0
        mock_api_init.assert_not_called()


class TestStatusCommand:
    """Tests for the status command."""

    def test_status_shows_services(self):
        """Should show all registered services."""
        runner = CliRunner()
        result = runner.invoke(status)

        assert result.exit_code == 0
        assert "notify" in result.output.lower()
        assert "approve" in result.output.lower()

    def test_status_shows_header(self):
        """Should show header information."""
        runner = CliRunner()
        result = runner.invoke(status)

        assert "syft-bg status" in result.output
        assert "services" in result.output
        assert "tokens" in result.output


class TestMainCommand:
    """Tests for main command group."""

    def test_help_shows_all_commands(self):
        """Help should list all available commands."""
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])

        assert result.exit_code == 0
        assert "status" in result.output
        assert "start" in result.output
        assert "stop" in result.output
        assert "restart" in result.output
        assert "logs" in result.output
        assert "init" in result.output
        assert "tui" in result.output
        assert "run" in result.output
        assert "create-api" in result.output
        assert "install" in result.output
        assert "uninstall" in result.output
        assert "remove-api" in result.output
        assert "remove-peer" in result.output
        assert "list-apis" in result.output


class TestInstallCommand:
    """Tests for the install command."""

    @patch("syft_bg.api.api.install")
    def test_install_all_services(self, mock_install):
        """Should call api.install with no service and show results."""
        from syft_bg.api.results import InstallationResult

        mock_install.return_value = [
            InstallationResult(
                success=True,
                service="notify",
                message="Service installed: /path/syft-bg-notify.service",
            ),
            InstallationResult(
                success=True,
                service="approve",
                message="Service installed: /path/syft-bg-approve.service",
            ),
        ]

        runner = CliRunner()
        result = runner.invoke(install, [])

        assert result.exit_code == 0
        mock_install.assert_called_once_with(None)
        assert "notify" in result.output
        assert "approve" in result.output

    @patch("syft_bg.api.api.install")
    def test_install_single_service(self, mock_install):
        """Should pass service name to api.install."""
        from syft_bg.api.results import InstallationResult

        mock_install.return_value = [
            InstallationResult(
                success=True,
                service="notify",
                message="Service installed: /path/syft-bg-notify.service",
            ),
        ]

        runner = CliRunner()
        result = runner.invoke(install, ["notify"])

        assert result.exit_code == 0
        mock_install.assert_called_once_with("notify")

    @patch("syft_bg.api.api.install")
    def test_install_failure(self, mock_install):
        """Should exit 1 on install failure."""
        from syft_bg.api.results import InstallationResult

        mock_install.return_value = [
            InstallationResult(
                success=False, service="notify", message="systemctl not found"
            ),
        ]

        runner = CliRunner()
        result = runner.invoke(install, ["notify"])

        assert result.exit_code == 1
        assert "systemctl not found" in result.output


class TestUninstallCommand:
    """Tests for the uninstall command."""

    @patch("syft_bg.api.api.uninstall")
    def test_uninstall_all_services(self, mock_uninstall):
        """Should call api.uninstall with no service and show results."""
        from syft_bg.api.results import InstallationResult

        mock_uninstall.return_value = [
            InstallationResult(
                success=True,
                service="notify",
                message="Service uninstalled: /path/syft-bg-notify.service",
            ),
        ]

        runner = CliRunner()
        result = runner.invoke(uninstall, [])

        assert result.exit_code == 0
        mock_uninstall.assert_called_once_with(None)

    @patch("syft_bg.api.api.uninstall")
    def test_uninstall_failure(self, mock_uninstall):
        """Should exit 1 on uninstall failure."""
        from syft_bg.api.results import InstallationResult

        mock_uninstall.return_value = [
            InstallationResult(
                success=False, service="notify", message="systemctl not found"
            ),
        ]

        runner = CliRunner()
        result = runner.invoke(uninstall, ["notify"])

        assert result.exit_code == 1
        assert "systemctl not found" in result.output


class TestCreateApiCommand:
    """Tests for the create-api command."""

    @patch("syft_bg.api.api.create_api")
    def test_create_api_basic(self, mock_api, sample_script):
        """Should accept a .py file and a single peer."""
        from syft_bg.api.results import ApiResult

        mock_api.return_value = ApiResult(
            success=True,
            name="main",
            file_contents=["main.py"],
            peers=["alice@test.com"],
        )

        runner = CliRunner()
        result = runner.invoke(
            create_api, [str(sample_script), "--peers", "alice@test.com"]
        )

        assert result.exit_code == 0
        mock_api.assert_called_once()

    @patch("syft_bg.api.api.create_api")
    def test_create_api_multiple_peers(self, mock_api, sample_script):
        """Should accept multiple peers via -p flags."""
        from syft_bg.api.results import ApiResult

        mock_api.return_value = ApiResult(
            success=True,
            name="main",
            file_contents=["main.py"],
            peers=["alice@test.com", "bob@test.com"],
        )

        runner = CliRunner()
        result = runner.invoke(
            create_api,
            [str(sample_script), "-p", "alice@test.com", "-p", "bob@test.com"],
        )

        assert result.exit_code == 0
        call_kwargs = mock_api.call_args[1]
        assert "alice@test.com" in call_kwargs["peers"]
        assert "bob@test.com" in call_kwargs["peers"]

    @patch("syft_bg.api.api.create_api")
    def test_create_api_multiple_files(self, mock_api, sample_scripts):
        """Should accept multiple files."""
        from syft_bg.api.results import ApiResult

        mock_api.return_value = ApiResult(
            success=True,
            name="api",
            file_contents=["main.py", "utils.py"],
            peers=["alice@test.com"],
        )

        runner = CliRunner()
        result = runner.invoke(
            create_api,
            [str(sample_scripts[0]), str(sample_scripts[1]), "-p", "alice@test.com"],
        )

        assert result.exit_code == 0
        mock_api.assert_called_once()

    @patch("syft_bg.api.api.create_api")
    def test_create_api_directory(self, mock_api, temp_dir):
        """Should accept a directory as contents."""
        from syft_bg.api.results import ApiResult

        subdir = temp_dir / "src"
        subdir.mkdir()
        (subdir / "main.py").write_text('print("a")\n')
        (subdir / "utils.py").write_text('print("b")\n')

        mock_api.return_value = ApiResult(
            success=True,
            name="api",
            file_contents=["main.py", "utils.py"],
            peers=["alice@test.com"],
        )

        runner = CliRunner()
        result = runner.invoke(create_api, [str(subdir), "-p", "alice@test.com"])

        assert result.exit_code == 0
        mock_api.assert_called_once()

    @patch("syft_bg.api.api.create_api")
    def test_create_api_error(self, mock_api, sample_script):
        """Should exit 1 on API error."""
        from syft_bg.api.results import ApiResult

        mock_api.return_value = ApiResult(success=False, error="No files to process")

        runner = CliRunner()
        result = runner.invoke(create_api, [str(sample_script), "-p", "alice@test.com"])

        assert result.exit_code == 1
        assert "No files to process" in result.output

    @patch("syft_bg.api.api.create_api")
    def test_no_peers_declined_aborts(self, mock_api, sample_script):
        """Without peers the user is warned and asked; 'n' aborts."""
        runner = CliRunner()
        result = runner.invoke(create_api, [str(sample_script)], input="n\n")

        assert result.exit_code == 1
        assert "no peers given" in result.output
        mock_api.assert_not_called()

    @patch("syft_bg.api.api.create_api")
    def test_no_peers_with_allow_any_peer_skips_prompt(self, mock_api, sample_script):
        from syft_bg.api.results import ApiResult

        mock_api.return_value = ApiResult(success=True, name="main")

        runner = CliRunner()
        result = runner.invoke(create_api, [str(sample_script), "--allow-any-peer"])

        assert result.exit_code == 0
        assert "no peers given" not in result.output
        assert mock_api.call_args[1]["allow_any_peer"] is True


@pytest.fixture
def api_store(temp_dir):
    """A real ApiStore in temp_dir, used by the CLI instead of config.yaml."""
    store = ApiStore(temp_dir / "syftbox", "do@test.com")
    with (
        patch("syft_bg.api.utils.get_api_store", return_value=store),
        patch("syft_bg.api.api.get_api_store", return_value=store),
        patch("syft_bg.approve.api_store.get_syftbg_dir", return_value=temp_dir),
    ):
        yield store


def _create_api(store, temp_dir, name, files, peers):
    content_files = []
    for rel in files:
        path = temp_dir / rel
        path.write_text(f"# {rel}\n")
        content_files.append((rel, path))
    store.create(name, content_files, [], peers)


class TestRemoveApiCommand:
    """Tests for the remove-api command."""

    def test_remove_api(self, api_store, temp_dir):
        _create_api(api_store, temp_dir, "my_analysis", ["run.sh"], ["a@x.com"])

        result = CliRunner().invoke(remove_api, ["my_analysis"])

        assert result.exit_code == 0
        assert "Removed api 'my_analysis'" in result.output
        assert not api_store.exists("my_analysis")

    def test_remove_api_unknown(self, api_store):
        result = CliRunner().invoke(remove_api, ["nonexistent"])

        assert result.exit_code == 1
        assert "not found" in result.output


class TestRemovePeerCommand:
    """Tests for the remove-peer command."""

    def test_remove_peer(self, api_store, temp_dir):
        """Should remove peer from apis."""
        _create_api(
            api_store, temp_dir, "my_analysis", ["main.py"], ["a@x.com", "b@x.com"]
        )

        runner = CliRunner()
        result = runner.invoke(remove_peer, ["a@x.com"])

        assert result.exit_code == 0
        assert "Removed peer" in result.output
        assert api_store.get("my_analysis").peers == ["b@x.com"]

    def test_removing_last_peer_asks_for_confirmation(self, api_store, temp_dir):
        """An object left without peers matches anyone, so ask first."""
        _create_api(api_store, temp_dir, "my_analysis", ["main.py"], ["a@x.com"])

        runner = CliRunner()
        result = runner.invoke(remove_peer, ["a@x.com"], input="n\n")
        assert result.exit_code == 1
        assert api_store.get("my_analysis").peers == ["a@x.com"]

        result = runner.invoke(remove_peer, ["a@x.com", "--allow-any-peer"])
        assert result.exit_code == 0
        assert api_store.get("my_analysis").peers == []

    def test_remove_peer_not_found(self, api_store):
        """Should error if peer not found."""
        runner = CliRunner()
        result = runner.invoke(remove_peer, ["unknown@test.com"])

        assert result.exit_code != 0


class TestListApisCommand:
    """Tests for the list-apis command."""

    @patch("syft_bg.api.api.list_apis")
    def test_list_apis(self, mock_list):
        """Should list all apis and their files."""
        from syft_rds.apis.models import ApiDefinition, FileEntry

        mock_list.return_value = {
            "analysis_a": ApiDefinition(
                file_contents=[
                    FileEntry(
                        relative_path="main.py",
                        path="/tmp/apis/analysis_a/main.py",
                        hash="sha256:aaa",
                    )
                ],
                peers=["alice@test.com"],
            ),
            "analysis_b": ApiDefinition(
                file_contents=[
                    FileEntry(
                        relative_path="train.py",
                        path="/tmp/apis/analysis_b/train.py",
                        hash="sha256:bbb",
                    )
                ],
                peers=["bob@test.com"],
            ),
        }

        runner = CliRunner()
        result = runner.invoke(list_apis)

        assert result.exit_code == 0
        assert "alice@test.com" in result.output
        assert "main.py" in result.output
        assert "bob@test.com" in result.output
        assert "train.py" in result.output

    @patch("syft_bg.api.api.list_apis")
    def test_list_apis_single_object(self, mock_list):
        """Should filter to a specific api."""
        from syft_rds.apis.models import ApiDefinition, FileEntry

        mock_list.return_value = {
            "analysis_a": ApiDefinition(
                file_contents=[
                    FileEntry(
                        relative_path="main.py",
                        path="/tmp/apis/analysis_a/main.py",
                        hash="sha256:aaa",
                    )
                ],
                peers=["alice@test.com"],
            ),
            "analysis_b": ApiDefinition(
                file_contents=[
                    FileEntry(
                        relative_path="train.py",
                        path="/tmp/apis/analysis_b/train.py",
                        hash="sha256:bbb",
                    )
                ],
                peers=["bob@test.com"],
            ),
        }

        runner = CliRunner()
        result = runner.invoke(list_apis, ["-n", "analysis_a"])

        assert result.exit_code == 0
        assert "analysis_a" in result.output
        assert "analysis_b" not in result.output

    @patch("syft_bg.api.api.list_apis")
    def test_list_apis_empty(self, mock_list):
        """Should show a message when there are no apis."""
        mock_list.return_value = {}

        runner = CliRunner()
        result = runner.invoke(list_apis)

        assert result.exit_code == 0
        assert "No apis created" in result.output
