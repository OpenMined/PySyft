# stdlib
import os
from pathlib import Path
from pathlib import PosixPath
import subprocess

# Make a log directory
log_path = Path("logs")
log_path.mkdir(exist_ok=True)

# Get the github job name and create a directory for it
job_name = os.getenv("GITHUB_JOB", "")
job_path: PosixPath = log_path / job_name
job_path.mkdir(exist_ok=True)

# Get all the containers running (per job)
containers = (
    subprocess.check_output("docker ps --format '{{.Names}}'", shell=True)
    .decode("utf-8")
    .split()
)

# Loop through the container ids and create a log file for each in the job directory
for container in containers:
    # Get the container name

    container_name = container.replace("'", "")

    # Get the container logs
    container_logs = subprocess.check_output(
        "docker logs " + container_name, shell=True, stderr=subprocess.STDOUT
    ).decode("utf-8")

    path = job_path / container_name
    path.write_text(container_logs, encoding="utf-8")

for file in job_path.iterdir():
    print(file)

print("============Log export completed for job: ", job_name)
