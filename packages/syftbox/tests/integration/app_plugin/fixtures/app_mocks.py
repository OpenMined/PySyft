import json
from pathlib import Path


class AppMockFactory:
    @staticmethod
    def create_app_without_config(apps_dir: Path, app_name: str) -> tuple[Path, str]:
        """Create a mock app without configuration."""
        mock_app_dir = Path(apps_dir) / app_name
        script_output = "Test executed"

        mock_app_dir.mkdir(parents=True, exist_ok=True)
        run_script = mock_app_dir / "run.sh"

        run_script.write_text(f"#!/bin/bash\necho '{script_output}' | tee app.log")
        run_script.chmod(0o755)

        return mock_app_dir, script_output

    @staticmethod
    def create_app_with_config(apps_dir: Path, app_name: str) -> tuple[Path, str]:
        """Create a mock app with configuration."""
        mock_app_dir = Path(apps_dir) / app_name
        mock_app_dir.mkdir(parents=True, exist_ok=True)

        test_value = "test_value"
        config_json = {
            "app": {
                "env": {"TEST_VAR": test_value},
                "run": {"schedule": None, "interval": 1, "command": ["./test.sh"]},
            }
        }

        config_file = mock_app_dir / "config.json"
        config_file.write_text(json.dumps(config_json))

        test_script = mock_app_dir / "test.sh"
        test_script.write_text("#!/bin/bash\necho $TEST_VAR | tee app.log")
        test_script.chmod(0o755)

        return mock_app_dir, test_value
