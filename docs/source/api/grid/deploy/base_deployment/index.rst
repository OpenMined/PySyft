:mod:`grid.deploy.base_deployment`
==================================

.. py:module:: grid.deploy.base_deployment


Module Contents
---------------

.. py:class:: BaseDeployment(env_vars, verbose: bool = True)

   Bases: :class:`abc.ABC`

   Abstract Class used to instantiate generic attributes for other deployment classes.

   .. method:: deploy(self)
      :abstractmethod:


      Method used to deploy component.


   .. method:: _check_dependency(self, lib='git', check='usage:', error_msg='Error: please install git.', verbose=False)


      This method checks if the environment has a specific dependency.
      :param dependency_lib: Libs that will be verified.
      :param check: Specific string to check if app was installed.
      :param error_msg: If not installed, raise an Exception with this.
      :param verbose: Used to define level of verbosity.

      :raises RuntimeError: If not installed, raise a RuntimeError Exception.


   .. method:: _execute(self, cmd)


      Execute a specific bash command and return the result.
      :param cmd: Specific bash command.

      :raises subprocess_exception: Raises an specific subprocess exception


   .. method:: _run_commands_in(self, commands, logs, tmp_dir='tmp', cleanup=True, verbose=False)


      Run sequentially all commands and logs stored in our list of commands/logs.
      :param commands: List of commands.
      :param logs: List of logs.
      :param tmp_dir: Directory used execute these commands.
      :param cleanup: Flag to choose if tmp_dir will be maintained.
      :param verbose: Used to define level of verbosity.

      :returns: Output message for each command.
      :rtype: outputs



