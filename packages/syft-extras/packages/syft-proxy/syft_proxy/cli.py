import os
import shutil
import subprocess
from typing import Tuple

import typer
import uvicorn

__version__ = "0.1.0"

# Constants
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9081
CERT_DIR = "./certs"
PROXY_DOMAIN = "syftbox.localhost"

app = typer.Typer(
    help="Syft Proxy Server CLI",
    add_completion=False,
    no_args_is_help=True,
)


@app.command()
def start(reload: bool = False) -> None:
    """Start the Syft Proxy Server."""

    cert_path, key_path = setup_https_certs()

    typer.echo(f"Starting Syft Proxy Server on https://{PROXY_DOMAIN}:{DEFAULT_PORT}")
    uvicorn.run(
        "syft_proxy.server:app",
        host=DEFAULT_HOST,
        port=DEFAULT_PORT,
        ssl_certfile=cert_path,
        ssl_keyfile=key_path,
        timeout_graceful_shutdown=5,
        timeout_keep_alive=10,
        workers=1,
        reload=reload,
    )


@app.command(name="bootstrap")
def bootstrap() -> None:
    """Initialize certificate chain and hosts file entries."""
    try:
        setup_cert_chain()
        typer.echo("✅ Setup self-signed cert chain")
        update_hosts_file()
        typer.echo("✅ Updated hosts file")
        typer.echo("✅ Bootstrap completed successfully")
    except Exception as e:
        typer.echo(f"❌ Bootstrap failed: {e}", err=True)
        raise typer.Exit(1)


def setup_cert_chain():
    try:
        subprocess.run(
            ["mkcert", "-install"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        typer.echo(f"❌ Failed to generate certificates: {e.stderr}")
        raise typer.Exit(1)


def setup_https_certs() -> Tuple[str, str]:
    """Generate HTTPS certificates using mkcert.

    Returns:
        Tuple[str, str]: Paths to the certificate and key files

    Raises:
        RuntimeError: If certificate generation fails
    """

    shutil.rmtree(CERT_DIR, ignore_errors=True)
    os.makedirs(CERT_DIR, exist_ok=True)
    cert_path = f"{CERT_DIR}/cert.pem"
    key_path = f"{CERT_DIR}/cert.key"

    try:
        subprocess.run(
            [
                "mkcert",
                "-install",
                "-cert-file",
                cert_path,
                "-key-file",
                key_path,
                PROXY_DOMAIN,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return (cert_path, key_path)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("Failed to generate HTTPS certificates") from e


def update_hosts_file() -> None:
    """Add syftbox.localhost entry to the hosts file.

    Raises:
        RuntimeError: If unable to modify the hosts file
    """
    hosts_path = (
        "/etc/hosts" if os.name != "nt" else r"C:\Windows\System32\drivers\etc\hosts"
    )
    entry = f"{DEFAULT_HOST} {PROXY_DOMAIN}"

    try:
        with open(hosts_path, "r") as f:
            content = f.read()
            if entry in content:
                return

        with open(hosts_path, "a") as f:
            f.write(f"\n{entry}\n")
    except PermissionError:
        raise RuntimeError("Insufficient permissions to modify hosts file")
    except Exception as e:
        raise RuntimeError("Failed to update hosts file") from e
