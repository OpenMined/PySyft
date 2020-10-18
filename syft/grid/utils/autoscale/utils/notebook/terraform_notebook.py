"""Helper methods to call terraform commands"""
import sys
import threading
import subprocess


def init():
    """
    args:
    """
    proc = subprocess.Popen(
        "/bin/sh",
        stdout=subprocess.PIPE,
        stdin=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    def outloop():
        running = True
        while running:
            line = proc.stdout.readline().decode(sys.stdout.encoding)
            print(line, end="")
            running = "\n" in line
        print("Exited")

    threading.Thread(target=outloop).start()

    commands = [b"terraform init\n", b"exit\n"]
    i = 0
    while proc.poll() is None and i < len(commands):
        inp = commands[i]
        if inp == "INPUT":
            inp = bytearray(input("") + "\n", sys.stdin.encoding)  # nosec
        if proc.poll() is None:
            proc.stdin.write(inp)
            proc.stdin.flush()
        i += 1


def apply():
    """
    args:
    """
    proc = subprocess.Popen(
        "/bin/sh",
        stdout=subprocess.PIPE,
        stdin=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    def outloop():
        running = True
        while running:
            line = proc.stdout.readline().decode(sys.stdout.encoding)
            print(line, end="")
            running = "\n" in line
        print("Exited")

    threading.Thread(target=outloop).start()

    commands = [b"terraform apply\n", "INPUT", b"exit\n"]
    i = 0
    while proc.poll() is None and i < len(commands):
        inp = commands[i]
        if inp == "INPUT":
            inp = bytearray(input("") + "\n", sys.stdin.encoding)  # nosec
        if proc.poll() is None:
            proc.stdin.write(inp)
            proc.stdin.flush()
        i += 1


def destroy():
    """
    args:
    """
    proc = subprocess.Popen(
        "/bin/sh",
        stdout=subprocess.PIPE,
        stdin=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    def outloop():
        running = True
        while running:
            line = proc.stdout.readline().decode(sys.stdout.encoding)
            print(line, end="")
            running = "\n" in line
        print("Exited")

    threading.Thread(target=outloop).start()

    commands = [b"terraform destroy\n", "INPUT", b"exit\n"]
    i = 0
    while proc.poll() is None and i < len(commands):
        inp = commands[i]
        if inp == "INPUT":
            inp = bytearray(input("") + "\n", sys.stdin.encoding)  # nosec
        if proc.poll() is None:
            proc.stdin.write(inp)
            proc.stdin.flush()
        i += 1
