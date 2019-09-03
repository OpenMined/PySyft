"""Utility functions."""
import os


def execute_command(command):
    """Executes the given command using the os inherent shell.

    Args:
        command (str): The command to execute
    """
    return os.popen(command).read()


def connect_all_nodes(nodes, sleep_time: float = 0.5):
    """Connect all nodes to each other.

    Args:
        nodes: A tuple of grid clients.
    """
    for i in range(len(nodes)):
        for j in range(i):
            node_i, node_j = nodes[i], nodes[j]
            node_i.connect_grid_node(node_j, sleep_time=sleep_time)
            node_j.connect_grid_node(node_i, sleep_time=sleep_time)
