:mod:`grid.deploy`
==================

.. py:module:: grid.deploy


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   base_deployment/index.rst
   heroku_gateway/index.rst
   heroku_node/index.rst


Package Contents
----------------

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



.. py:class:: HerokuGatewayDeployment(app_name: str, verbose=True, check_deps=True, dev_user: str = 'OpenMined', branch: set = 'dev', env_vars={})

   Bases: :class:`grid.deploy.BaseDeployment`

   An abstraction of heroku grid gateway deployment process, the purpose of this class is set all configuration needed to deploy grid gateway application in heroku platform.

   .. method:: deploy(self)


      Method to deploy Grid Gateway app on heroku platform.


   .. method:: __run_heroku_commands(self)


      Add a set of commands/logs used to deploy grid gateway app on heroku platform.


   .. method:: __check_heroku_dependencies(self)


      Check specific dependencies to perform grid gateway deploy on heroku platform.



