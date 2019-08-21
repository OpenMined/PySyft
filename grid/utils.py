"""Utility functions."""
import os


def execute_command(command):
    return os.popen(command).read()
