# stdlib
import os
from pathlib import Path
import subprocess

# Make a log directory
log_path = Path("log")
log_path.mkdir(exist_ok=True)

# Get the github job name and create a directory for it
job_name = os.getenv("GITHUB_JOB")
job_path = log_path / job_name
job_path.mkdir(exist_ok=True)

# Get all the containers running (per job)
containers = (
    subprocess.check_output("docker ps -a -q", shell=True).decode("utf-8").split()
)

# Loop through the container ids and create a log file for each in the job directory
for container in containers:
    # Get the container name
    container_name = (
        subprocess.check_output(
            "docker inspect --format '{{.Name}}' " + container, shell=True
        )
        .decode("utf-8")
        .strip()
    )

    # Get the container logs
    container_logs = subprocess.check_output(
        "docker logs " + container, shell=True
    ).decode("utf-8")

    unquoted_name = container_name.replace("'", "")
    path = job_path / unquoted_name
    path.write_text(container_logs)

print("============Log export completed for job: ", job_name)
