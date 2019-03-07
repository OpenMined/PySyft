import subprocess


def kill_command(pid):
    cmd = "kill " + pid
    process = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    # wait for the process to terminate
    out, err = process.communicate()
    errcode = process.returncode


cmd = "ps -ef | grep flask"
process = subprocess.Popen(
    cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
)

# wait for the process to terminate
out, err = process.communicate()
for row in list(
    filter(
        lambda x: "SCREEN" not in x
        and "grep" not in x
        and "kill" not in x
        and "run" in x,
        out.decode("ascii").split("\n"),
    )
):

    try:
        pid = str(int(row.split(" ")[1]))
        kill_command(pid)
    except:
        ""

cmd = "ps -ef | grep create_process"
process = subprocess.Popen(
    cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
)

# wait for the process to terminate
out, err = process.communicate()
for row in list(
    filter(lambda x: "SCREEN" in x and "grep" not in x, out.decode("ascii").split("\n"))
):
    pid = str(int(row.split(" ")[1]) + 1)
    kill_command(pid)
