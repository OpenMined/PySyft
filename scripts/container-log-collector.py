# stdlib
import os
import subprocess

# Make a log directory
cwd = os.getcwd()
log_path = os.path.join(cwd, "logs")
if not os.path.exists(log_path):
    os.makedirs(log_path)

# Get the github job name and create a directory for it
job_name = "test"  # os.getenv("GITHUB_JOB")
job_path = os.path.join(log_path, job_name)
if not os.path.exists(job_path):
    os.makedirs(job_path)

# Get all the containers running (per job)
containers = (
    subprocess.check_output("docker ps -a -q", shell=True).decode("utf-8").split()
)

# Loop through the container ids and create a log file for each in the job directory
print("============Log export started for job: ", job_name)
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

    # Store container logs in a file
    with open(f"{job_path}{container_name}.log", "w") as f:
        f.write(container_logs)
        f.close()
print("============Log export completed for job: ", job_name)
