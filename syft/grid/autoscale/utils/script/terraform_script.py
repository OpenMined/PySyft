"""Helper methods to call terrafrom commands"""
import subprocess


def init():
    """
    args:
    """
    subprocess.call("terraform init", shell=True)


def apply():
    """
    args:
    """
    subprocess.call("terraform apply", shell=True)


def destroy():
    """
    args:
    """
    subprocess.call("terraform destroy", shell=True)
