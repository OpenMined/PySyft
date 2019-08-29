"""Utility functions."""
import os


def execute_command(command):
    """Executes the given command using the os inherent shell.

        Args:
            command (str): The command to execute
    """
    return os.popen(command).read()
