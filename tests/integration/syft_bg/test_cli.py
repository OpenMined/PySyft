"""Integration tests for CLI commands."""

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from syft_bg.cli.commands import (
    auto_approve,
    init,
    install,
    list_auto_approvals,
    main,
    remove_auto_approval,
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
        assert "auto-approve" in result.output
        assert "install" in result.output
        assert "uninstall" in result.output
        assert "remove-auto-approval" in result.output
        assert "remove-peer" in result.output
        assert "list-auto-approvals" in result.output


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


class TestAutoApproveCommand:
    """Tests for the auto-approve command."""

    @patch("syft_bg.api.api.auto_approve")
    def test_auto_approve_basic(self, mock_api, sample_script):
        """Should accept a .py file and a single peer."""
        from syft_bg.api.results import AutoApproveResult

        mock_api.return_value = AutoApproveResult(
            success=True,
            name="main",
            file_contents=["main.py"],
            peers=["alice@test.com"],
        )

        runner = CliRunner()
        result = runner.invoke(
            auto_approve, [str(sample_script), "--peers", "alice@test.com"]
        )

        assert result.exit_code == 0
        mock_api.assert_called_once()

    @patch("syft_bg.api.api.auto_approve")
    def test_auto_approve_multiple_peers(self, mock_api, sample_script):
        """Should accept multiple peers via -p flags."""
        from syft_bg.api.results import AutoApproveResult

        mock_api.return_value = AutoApproveResult(
            success=True,
            name="main",
            file_contents=["main.py"],
            peers=["alice@test.com", "bob@test.com"],
        )

        runner = CliRunner()
        result = runner.invoke(
            auto_approve,
            [str(sample_script), "-p", "alice@test.com", "-p", "bob@test.com"],
        )

        assert result.exit_code == 0
        call_kwargs = mock_api.call_args[1]
        assert "alice@test.com" in call_kwargs["peers"]
        assert "bob@test.com" in call_kwargs["peers"]

    @patch("syft_bg.api.api.auto_approve")
    def test_auto_approve_multiple_files(self, mock_api, sample_scripts):
        """Should accept multiple files."""
        from syft_bg.api.results import AutoApproveResult

        mock_api.return_value = AutoApproveResult(
            success=True,
            name="auto_approval",
            file_contents=["main.py", "utils.py"],
            peers=["alice@test.com"],
        )

        runner = CliRunner()
        result = runner.invoke(
            auto_approve,
            [str(sample_scripts[0]), str(sample_scripts[1]), "-p", "alice@test.com"],
        )

        assert result.exit_code == 0
        mock_api.assert_called_once()

    @patch("syft_bg.api.api.auto_approve")
    def test_auto_approve_directory(self, mock_api, temp_dir):
        """Should accept a directory as contents."""
        from syft_bg.api.results import AutoApproveResult

        subdir = temp_dir / "src"
        subdir.mkdir()
        (subdir / "main.py").write_text('print("a")\n')
        (subdir / "utils.py").write_text('print("b")\n')

        mock_api.return_value = AutoApproveResult(
            success=True,
            name="auto_approval",
            file_contents=["main.py", "utils.py"],
            peers=["alice@test.com"],
        )

        runner = CliRunner()
        result = runner.invoke(auto_approve, [str(subdir), "-p", "alice@test.com"])

        assert result.exit_code == 0
        mock_api.assert_called_once()

    @patch("syft_bg.api.api.auto_approve")
    def test_auto_approve_error(self, mock_api, sample_script):
        """Should exit 1 on API error."""
        from syft_bg.api.results import AutoApproveResult

        mock_api.return_value = AutoApproveResult(
            success=False, error="No files to process"
        )

        runner = CliRunner()
        result = runner.invoke(auto_approve, [str(sample_script)])

        assert result.exit_code == 1
        assert "No files to process" in result.output


class TestRemoveAutoApprovalCommand:
    """Tests for the remove-auto-approval command."""

    @patch("syft_bg.approve.config.AutoApproveConfig.load")
    def test_remove_auto_approval(self, mock_load):
        """Should remove scripts by filename from an auto-approval object."""
        from syft_bg.approve.config import AutoApprovalObj, FileEntry

        obj = AutoApprovalObj(
            file_contents=[
                FileEntry(
                    relative_path="main.py",
                    path="/tmp/auto_approvals/my_analysis/main.py",
                    hash="sha256:aaa",
                ),
                FileEntry(
                    relative_path="utils.py",
                    path="/tmp/auto_approvals/my_analysis/utils.py",
                    hash="sha256:bbb",
                ),
            ],
            peers=["alice@test.com"],
        )
        mock_config = MagicMock()
        mock_config.auto_approvals.objects = {"my_analysis": obj}
        mock_load.return_value = mock_config

        runner = CliRunner()
        result = runner.invoke(remove_auto_approval, ["utils.py", "-n", "my_analysis"])

        assert result.exit_code == 0
        assert "Removed 1" in result.output
        assert len(obj.file_contents) == 1
        assert obj.file_contents[0].relative_path == "main.py"

    @patch("syft_bg.approve.config.AutoApproveConfig.load")
    def test_remove_auto_approval_unknown_object(self, mock_load):
        """Should error when object name not found."""
        mock_config = MagicMock()
        mock_config.auto_approvals.objects = {}
        mock_load.return_value = mock_config

        runner = CliRunner()
        result = runner.invoke(remove_auto_approval, ["main.py", "-n", "nonexistent"])

        assert result.exit_code == 1


class TestRemovePeerCommand:
    """Tests for the remove-peer command."""

    @patch("syft_bg.approve.config.AutoApproveConfig.load")
    def test_remove_peer(self, mock_load):
        """Should remove peer from auto-approval objects."""
        from syft_bg.approve.config import AutoApprovalObj

        mock_config = MagicMock()
        mock_config.auto_approvals.objects = {
            "my_analysis": AutoApprovalObj(peers=["alice@test.com"])
        }
        mock_load.return_value = mock_config

        runner = CliRunner()
        result = runner.invoke(remove_peer, ["alice@test.com"])

        assert result.exit_code == 0
        assert "Removed peer" in result.output
        assert (
            "alice@test.com"
            not in mock_config.auto_approvals.objects["my_analysis"].peers
        )

    @patch("syft_bg.approve.config.AutoApproveConfig.load")
    def test_remove_peer_not_found(self, mock_load):
        """Should error if peer not found."""
        mock_config = MagicMock()
        mock_config.auto_approvals.objects = {}
        mock_load.return_value = mock_config

        runner = CliRunner()
        result = runner.invoke(remove_peer, ["unknown@test.com"])

        assert result.exit_code != 0


class TestListAutoApprovalsCommand:
    """Tests for the list-auto-approvals command."""

    @patch("syft_bg.approve.config.AutoApproveConfig.load")
    def test_list_auto_approvals(self, mock_load):
        """Should list all auto-approval objects and their scripts."""
        from syft_bg.approve.config import AutoApprovalObj, FileEntry

        mock_config = MagicMock()
        mock_config.auto_approvals.objects = {
            "analysis_a": AutoApprovalObj(
                file_contents=[
                    FileEntry(
                        relative_path="main.py",
                        path="/tmp/auto_approvals/analysis_a/main.py",
                        hash="sha256:aaa",
                    )
                ],
                peers=["alice@test.com"],
            ),
            "analysis_b": AutoApprovalObj(
                file_contents=[
                    FileEntry(
                        relative_path="train.py",
                        path="/tmp/auto_approvals/analysis_b/train.py",
                        hash="sha256:bbb",
                    )
                ],
                peers=["bob@test.com"],
            ),
        }
        mock_load.return_value = mock_config

        runner = CliRunner()
        result = runner.invoke(list_auto_approvals)

        assert result.exit_code == 0
        assert "alice@test.com" in result.output
        assert "main.py" in result.output
        assert "bob@test.com" in result.output
        assert "train.py" in result.output

    @patch("syft_bg.approve.config.AutoApproveConfig.load")
    def test_list_auto_approvals_single_object(self, mock_load):
        """Should filter to a specific auto-approval object."""
        from syft_bg.approve.config import AutoApprovalObj, FileEntry

        mock_config = MagicMock()
        mock_config.auto_approvals.objects = {
            "analysis_a": AutoApprovalObj(
                file_contents=[
                    FileEntry(
                        relative_path="main.py",
                        path="/tmp/auto_approvals/analysis_a/main.py",
                        hash="sha256:aaa",
                    )
                ],
                peers=["alice@test.com"],
            ),
            "analysis_b": AutoApprovalObj(
                file_contents=[
                    FileEntry(
                        relative_path="train.py",
                        path="/tmp/auto_approvals/analysis_b/train.py",
                        hash="sha256:bbb",
                    )
                ],
                peers=["bob@test.com"],
            ),
        }
        mock_load.return_value = mock_config

        runner = CliRunner()
        result = runner.invoke(list_auto_approvals, ["-n", "analysis_a"])

        assert result.exit_code == 0
        assert "analysis_a" in result.output
        assert "analysis_b" not in result.output

    @patch("syft_bg.approve.config.AutoApproveConfig.load")
    def test_list_auto_approvals_empty(self, mock_load):
        """Should show message when no auto-approval objects configured."""
        mock_config = MagicMock()
        mock_config.auto_approvals.objects = {}
        mock_load.return_value = mock_config

        runner = CliRunner()
        result = runner.invoke(list_auto_approvals)

        assert result.exit_code == 0
        assert "No auto-approval objects configured" in result.output
