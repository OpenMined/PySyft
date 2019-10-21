"""Utility functions."""
import os


def execute_command(command: str) -> str:
    """Executes the given command using the os inherent shell.

    Args:
        command (str): The command to execute
    Returns:
        result (str) : command's result
    """
    return os.popen(command).read()


def connect_all_nodes(nodes: list):
    """Connect all nodes to each other.

    Args:
        nodes: A tuple of grid clients.
    """
    for i in range(len(nodes)):
        for j in range(i):
            node_i, node_j = nodes[i], nodes[j]
            node_i.connect_nodes(node_j)
            node_j.connect_nodes(node_i)
