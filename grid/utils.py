"""Utility functions."""
import os

def exec_os_cmd(command):
    return os.popen(command).read()