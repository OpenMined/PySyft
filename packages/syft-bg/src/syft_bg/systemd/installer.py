"""Systemd user service installer for syft-bg."""

import subprocess
import sys
from pathlib import Path

SERVICE_PREFIX = "syft-bg"

SERVICE_TEMPLATE = """\
[Unit]
Description=SyftBox Background Service — {service}
After=network-online.target
Wants=network-online.target

[Service]
Type=forking
ExecStart={python} -m syft_bg start {service}
ExecStop={python} -m syft_bg stop {service}
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
"""


def get_systemd_user_dir() -> Path:
    """Get the systemd user service directory."""
    return Path.home() / ".config" / "systemd" / "user"


def _service_filename(service: str) -> str:
    return f"{SERVICE_PREFIX}-{service}.service"


def get_service_path(service: str) -> Path:
    """Get the full path to a service's systemd unit file."""
    return get_systemd_user_dir() / _service_filename(service)


def is_installed(service: str) -> bool:
    """Check whether a service's systemd unit file exists."""
    return get_service_path(service).exists()


def _daemon_reload() -> tuple[bool, str]:
    # systemd caches unit files in memory; reload forces it to re-read from disk.
    # needed before `enable` so systemd can find the newly written unit file.
    result = subprocess.run(
        ["systemctl", "--user", "daemon-reload"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return (False, f"Failed to reload systemd: {result.stderr}")
    return (True, "")


def install_service(service: str) -> tuple[bool, str]:
    """Install a single syft-bg service as a systemd user unit.

    Returns:
        (success, message) tuple
    """
    service_dir = get_systemd_user_dir()
    service_path = get_service_path(service)
    unit_name = _service_filename(service).removesuffix(".service")

    try:
        service_dir.mkdir(parents=True, exist_ok=True)

        python_path = sys.executable
        service_content = SERVICE_TEMPLATE.format(python=python_path, service=service)
        service_path.write_text(service_content)

        ok, err = _daemon_reload()
        if not ok:
            service_path.unlink(missing_ok=True)
            return (False, err)

        result = subprocess.run(
            ["systemctl", "--user", "enable", unit_name],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            service_path.unlink(missing_ok=True)
            return (False, f"Failed to enable service: {result.stderr}")

        return (True, f"Service installed: {service_path}")

    except PermissionError:
        return (False, f"Permission denied writing to {service_path}")
    except FileNotFoundError:
        service_path.unlink(missing_ok=True)
        return (False, "systemctl not found — systemd may not be available")
    except Exception as e:
        service_path.unlink(missing_ok=True)
        return (False, str(e))


def uninstall_service(service: str) -> tuple[bool, str]:
    """Uninstall a single syft-bg service's systemd user unit.

    Returns:
        (success, message) tuple
    """
    service_path = get_service_path(service)
    unit_name = _service_filename(service).removesuffix(".service")

    try:
        subprocess.run(
            ["systemctl", "--user", "stop", unit_name],
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["systemctl", "--user", "disable", unit_name],
            capture_output=True,
            text=True,
        )

        if service_path.exists():
            service_path.unlink()

        _daemon_reload()

        return (True, f"Service uninstalled: {service_path}")

    except FileNotFoundError:
        return (False, "systemctl not found — systemd may not be available")
    except Exception as e:
        return (False, str(e))
