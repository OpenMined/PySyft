:mod:`grid.utils`
=================

.. py:module:: grid.utils

.. autoapi-nested-parse::

   Utility functions.



Module Contents
---------------

.. function:: execute_command(command: str) -> str

   Executes the given command using the os inherent shell.

   :param command: The command to execute
   :type command: str

   :returns: command's result
   :rtype: result (str)


.. function:: connect_all_nodes(nodes: list)

   Connect all nodes to each other.

   :param nodes: A tuple of grid clients.


