import hashlib
import json
import logging
import os
import shutil
import subprocess
import threading
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from subprocess import CompletedProcess
from types import SimpleNamespace

from croniter import croniter
from loguru import logger
from typing_extensions import Any, Optional, Union

from syftbox.client.base import SyftBoxContextInterface
from syftbox.lib.client_config import CONFIG_PATH_ENV
from syftbox.lib.types import PathLike

APP_LOG_FILE_NAME_FORMAT = "{app_name}.log"
DEFAULT_INTERVAL = 10
RUNNING_APPS: dict = {}
DEFAULT_APPS_PATH = Path(os.path.join(os.path.dirname(__file__), "..", "..", "..", "default_apps")).absolute().resolve()
EVENT = threading.Event()


def path_without_virtualenvs() -> str:
    env_path = os.getenv("PATH", "")
    if not env_path:
        return env_path

    venv_hints = [
        f"env{os.sep}bin",
        f"env{os.sep}Scripts",
        "conda",
        ".virtualenvs",
        "pyenv",
    ]

    # activated venv will have VIRTUAL_ENV and VIRTUAL_ENV/bin in PATH
    # so axe it
    env_venv = os.getenv("VIRTUAL_ENV", "")
    if env_venv:
        venv_hints.append(env_venv)

    cleaned_path = [
        entry for entry in env_path.split(os.pathsep) if not any(hint in entry.lower() for hint in venv_hints)
    ]

    return os.pathsep.join(cleaned_path)


def get_clean_env() -> dict:
    clean_env: dict = {}

    essential_vars = {
        "PATH",
        "HOME",
        "USER",
        "TEMP",
        "TMP",
        "TMPDIR",
        "SHELL",
        "LANG",
        "LC_ALL",
        "DISPLAY",  # X11 specific (Linux)
        "DBUS_SESSION_BUS_ADDRESS",  # X11 specific (Linux)
        "SYSTEMROOT",  # Windows specific
    }

    # Copy essential and SYFTBOX_* variables
    for key, value in os.environ.items():
        if key in essential_vars or key.startswith("SYFTBOX_"):
            clean_env[key] = value

    return clean_env


def find_and_run_script(
    app_path: Path, extra_args: list[str], config_path: Path, app_log_dir: Optional[Path] = None
) -> CompletedProcess[str]:
    script_path = os.path.join(app_path, "run.sh")

    clean_env = get_clean_env()
    clean_env.update(
        {
            "PATH": path_without_virtualenvs(),
            CONFIG_PATH_ENV: str(config_path),
        }
    )

    # Check if the script exists
    if os.path.isfile(script_path):
        # Set execution bit (+x)
        os.chmod(script_path, os.stat(script_path).st_mode | 0o111)

        # Prepare the command based on whether there's a shebang or not
        command = ["sh", script_path] + extra_args

        result, _ = run_with_logging(
            command,
            app_path,
            clean_env,
            app_log_dir,
        )
        return result
    else:
        raise FileNotFoundError(f"run.sh not found in {app_path}")


def create_app_logger(log_file: Path) -> tuple[logging.Logger, RotatingFileHandler]:
    """Create an isolated logger for app runs"""
    # Create a new logger instance
    logger = logging.getLogger(f"app_logger_{log_file.name}")
    logger.setLevel(logging.INFO)

    # Remove any existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d at %H:%M:%S")

    # Create and configure file handler
    file_handler = RotatingFileHandler(
        filename=log_file,
        maxBytes=100 * 1024 * 1024,  # 100Mb
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    # Add handler to logger
    logger.addHandler(file_handler)

    return logger, file_handler


def run_with_logging(
    command: list[str], app_path: Path, clean_env: dict, log_path: Optional[Path] = None
) -> tuple[CompletedProcess[str], Path]:
    """
    Run a subprocess command and capture output to both a log file and return results.
    """
    # Create logs directory if it doesn't exist
    if log_path is None:
        log_path = app_path / "logs"
    log_path.mkdir(parents=True, exist_ok=True)

    # Create a unique log filename with timestamp and app name
    app_name = app_path.name
    log_file = log_path / APP_LOG_FILE_NAME_FORMAT.format(app_name=app_name)

    # Create isolated logger for this run
    app_logger, file_handler = create_app_logger(log_file=log_file)

    try:
        # Log run metadata
        app_logger.info(f"Working directory: {app_path}")
        app_logger.info(f"Command: {command}")

        # Run the subprocess
        process = subprocess.run(
            command,
            cwd=app_path,
            check=True,
            capture_output=True,
            text=True,
            env=clean_env,
        )

        # log this in app log
        app_logger.info(
            (
                f"exit code: {process.returncode}\n"
                f"===stdout===\n{process.stdout}"
                f"===stderr===\n{process.stderr or '-'}"
            )
        )
        # log this in console
        logger.info(f"Process completed with exit code: {process.returncode}. Log file: {log_file}")
        return process, log_file

    except subprocess.CalledProcessError as e:
        # log this in app log
        app_logger.error(
            ("process output\n" f"> exit code: {e.returncode}\n" f"> stdout:\n{e.stdout}" f"> stderr:\n{e.stderr}")
        )
        # log this in console
        logger.error(f"Process failed with exit code: {e.returncode}")
        raise e

    except Exception as e:
        app_logger.error(f"Unexpected error: {str(e)}")
        raise e

    finally:
        app_logger.removeHandler(file_handler)
        file_handler.close()


def copy_default_apps(apps_path: Path) -> None:
    if not DEFAULT_APPS_PATH.exists():
        logger.info(f"Default apps directory not found: {DEFAULT_APPS_PATH}")
        return

    for app in DEFAULT_APPS_PATH.iterdir():
        src_app_path = DEFAULT_APPS_PATH / app
        dst_app_path = apps_path / app.name

        if src_app_path.is_dir():
            if dst_app_path.exists():
                logger.info(f"App already installed at: {dst_app_path}")
                # shutil.rmtree(dst_app_path)
            else:
                shutil.copytree(src_app_path, dst_app_path)
                logger.info(f"Copied default app:: {app}")


def dict_to_namespace(data: Union[dict, list, Any]) -> Union[SimpleNamespace, list, Any]:
    if isinstance(data, dict):
        return SimpleNamespace(**{key: dict_to_namespace(value) for key, value in data.items()})
    elif isinstance(data, list):
        return [dict_to_namespace(item) for item in data]
    else:
        return data


def load_config(path: PathLike) -> Optional[Union[SimpleNamespace, list, Any]]:
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return dict_to_namespace(data)
    except Exception:
        return None


def bootstrap(context: SyftBoxContextInterface) -> None:
    # create the directory
    apps_path = context.workspace.apps

    apps_path.mkdir(exist_ok=True)

    # Copy default apps if they don't exist
    copy_default_apps(apps_path)


def run_apps(apps_path: Path, client_config: Path) -> None:
    # create the directory

    for app in apps_path.iterdir():
        app_path = apps_path.absolute() / app
        if app_path.is_dir():
            app_config = load_config(app_path / "config.json")
            if app_config is None:
                run_app(app_path, client_config)
            elif RUNNING_APPS.get(app, None) is None:
                logger.info("⏱  Scheduling a  new app run.")
                thread = threading.Thread(
                    target=run_custom_app_config,
                    args=(
                        app_config,
                        app_path,
                        client_config,
                    ),
                )
                thread.start()
                RUNNING_APPS[os.path.basename(app)] = thread


def get_file_hash(file_path: Union[str, Path], digest: str = "md5") -> str:
    with open(file_path, "rb") as f:
        return hashlib.file_digest(f, digest).hexdigest()


def output_published(app_output: Union[str, Path], published_output: Union[str, Path]) -> bool:
    return (
        os.path.exists(app_output)
        and os.path.exists(published_output)
        and get_file_hash(app_output, "md5") == get_file_hash(published_output, "md5")
    )


def run_custom_app_config(app_config: SimpleNamespace, app_path: Path, client_config: Path) -> None:
    app_name = os.path.basename(app_path)
    clean_env = {
        "PATH": path_without_virtualenvs(),
        CONFIG_PATH_ENV: str(client_config),
    }
    # Update environment with any custom variables in app_config
    app_envs = getattr(app_config.app, "env", {})
    if not isinstance(app_envs, dict):
        app_envs = vars(app_envs)
    clean_env.update(app_envs)

    # Retrieve the cron-style schedule from app_config
    cron_iter = None
    interval = None
    cron_schedule = getattr(app_config.app.run, "schedule", None)
    if cron_schedule is not None:
        base_time = datetime.now()
        cron_iter = croniter(cron_schedule, base_time)
    elif getattr(app_config.app.run, "interval", None) is not None:
        raw_interval = app_config.app.run.interval
        if not isinstance(raw_interval, (int, float)):
            raise ValueError(f"Invalid interval type: {type(raw_interval)}. Expected int or float.")
        interval = raw_interval
    else:
        raise Exception("There's no schedule configuration. Please add schedule or interval in your app config.json")

    while not EVENT.is_set():
        current_time = datetime.now()
        logger.info(f"👟 Running {app_name} at scheduled time {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Running command: {app_config.app.run.command}")
        try:
            app_log_dir = app_path / "logs"
            run_with_logging(
                app_config.app.run.command,
                app_path,
                clean_env,
                app_log_dir,
            )
            log_file = app_log_dir / APP_LOG_FILE_NAME_FORMAT.format(app_name=app_name)
            logger.info(f"App '{app_name}' ran successfully. \nDetailed logs at: {log_file.resolve()}")
        except subprocess.CalledProcessError as _:
            logger.error(f"Error calling subprocess for api '{app_name}'")
            logger.error(f"Check {app_name}'s api logs at: {log_file.resolve()}")
        except Exception as _:
            logger.error(f"Error running '{app_name}'")
            logger.error(f"Check {app_name} api logs at: {log_file.resolve()}")

        if cron_iter is not None:
            # Schedule the next execution
            next_execution = cron_iter.get_next(datetime)
            time_to_wait = int((next_execution - current_time).total_seconds())
            logger.info(
                f"⏲ Waiting for scheduled time. Current time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}, Next execution: {next_execution.strftime('%Y-%m-%d %H:%M:%S')}"
            )
        else:
            if interval is not None:
                time_to_wait = int(interval)
        time.sleep(time_to_wait)


def run_app(app_path: Path, config_path: Path) -> None:
    app_name = os.path.basename(app_path)
    app_log_dir = app_path / "logs"
    log_file = app_log_dir / APP_LOG_FILE_NAME_FORMAT.format(app_name=app_name)

    extra_args: list = []
    try:
        logger.info(f"Running '{app_name}' app")
        find_and_run_script(app_path, extra_args, config_path, app_log_dir)
        logger.info(f"`{app_name}` App ran successfully. \nDetailed logs at: {log_file.resolve()}")
    except FileNotFoundError as e:
        logger.error(f"Error running '{app_name}'")
        logger.error(f"Error: {str(e)}")
    except subprocess.CalledProcessError as _:
        logger.error(f"Error calling subprocess for api '{app_name}'")
        logger.error(f"Check {app_name}'s api logs at: {log_file.resolve()}")
    except Exception as _:
        logger.error(f"Error running '{app_name}'")
        logger.error(f"Check {app_name} api logs at: {log_file.resolve()}")


class AppRunner:
    def __init__(self, context: SyftBoxContextInterface, interval: int = DEFAULT_INTERVAL):
        self.context = context
        self.__event = threading.Event()
        self.interval = interval
        self.__run_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        def run() -> None:
            bootstrap(self.context)
            while not self.__event.is_set():
                try:
                    run_apps(
                        apps_path=self.context.workspace.apps,
                        client_config=self.context.config.path,
                    )
                    self.__event.wait(self.interval)
                except Exception as e:
                    logger.error(f"Error running apps: {str(e)}")

        self.__run_thread = threading.Thread(target=run)
        self.__run_thread.start()

    def stop(self, blocking: bool = False) -> None:
        if not self.__run_thread:
            return

        EVENT.set()
        self.__event.set()
        if blocking:
            self.__run_thread.join()
