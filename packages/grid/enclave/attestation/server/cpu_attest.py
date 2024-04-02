# stdlib
import subprocess

# third party
from loguru import logger


def cpu_attest() -> str:
    # Fetch report from Micrsoft Attestation library
    cpu_report = subprocess.run(
        ["/app/AttestationClient"], capture_output=True, text=True
    )
    logger.info(f"Stdout: {cpu_report.stdout}")
    logger.info(f"Stderr: {cpu_report.stderr}")
    if cpu_report.returncode == 0 and cpu_report.stdout == "true":
        return "True"

    return "False"
