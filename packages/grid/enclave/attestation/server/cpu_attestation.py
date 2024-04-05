# stdlib
import subprocess

# third party
from loguru import logger


def attest_cpu() -> str:
    # Fetch report from Micrsoft Attestation library
    cpu_report = subprocess.run(
        ["/app/AttestationClient"], capture_output=True, text=True
    )
    logger.debug(f"Stdout: {cpu_report.stdout}")
    logger.debug(f"Stderr: {cpu_report.stderr}")

    logger.info("Attestation Return Code: {}", cpu_report.returncode)
    if cpu_report.returncode == 0 and cpu_report.stdout == "true":
        return "True"

    return "False"


def get_cpu_token() -> str:
    # Fetch token from Micrsoft Attestation library
    cpu_token = subprocess.run(
        ["/app/AttestationClient", "-o token"], capture_output=True, text=True
    )
    logger.debug(f"Stdout: {cpu_token.stdout}")
    logger.debug(f"Stderr: {cpu_token.stderr}")

    logger.info("Attestation Token Return Code: {}", cpu_token.returncode)
    return cpu_token.stdout
