# Loop through all containers and prinnt container id and container name using subprocess module
# stdlib
import subprocess

# Get all containers
containers = (
    subprocess.check_output("docker ps -a -q", shell=True).decode("utf-8").split()
)
# print("Containers: ", containers)

# Store logs of each container in a file
for container in containers:
    # Get container name
    container_name = (
        subprocess.check_output(
            "docker inspect --format '{{.Name}}' " + container, shell=True
        )
        .decode("utf-8")
        .strip()
    )
    print("Container Name: ", container_name)
    # Get container logs
    container_logs = subprocess.check_output(
        "docker logs " + container, shell=True
    ).decode("utf-8")

    # Store container logs in a file
    with open(f"./{container_name}.log", "w") as f:
        f.write(container_logs)
        f.close()
