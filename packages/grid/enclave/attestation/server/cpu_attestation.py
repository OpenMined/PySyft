# stdlib
import subprocess

# third party
from loguru import logger


def attest_cpu() -> tuple[str, str]:
    # Fetch report from Micrsoft Attestation library
    cpu_report = subprocess.run(
        ["/app/AttestationClient"], capture_output=True, text=True
    )
    logger.debug(f"Stdout: {cpu_report.stdout}")
    logger.debug(f"Stderr: {cpu_report.stderr}")

    logger.info("Attestation Return Code: {}", cpu_report.returncode)
    res = "False"
    if cpu_report.returncode == 0 and cpu_report.stdout == "true":
        res = "True"

    # Fetch token from Micrsoft Attestation library
    cpu_token = subprocess.run(
        ["/app/AttestationClient", "-o", "token"], capture_output=True, text=True
    )
    logger.debug(f"Stdout: {cpu_token.stdout}")
    logger.debug(f"Stderr: {cpu_token.stderr}")

    logger.info("Attestation Token Return Code: {}", cpu_token.returncode)
    return res, cpu_token.stdout
