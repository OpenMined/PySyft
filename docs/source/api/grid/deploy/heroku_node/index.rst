:mod:`grid.deploy.heroku_node`
==============================

.. py:module:: grid.deploy.heroku_node


Module Contents
---------------

.. py:class:: HerokuNodeDeployment(grid_name: str, verbose=True, check_deps=True, app_type: str = 'websocket', dev_user: str = 'OpenMined', branch: set = 'dev', env_vars={})

   Bases: :class:`grid.deploy.BaseDeployment`

   An abstraction of heroku grid node deployment process, the purpose of this class is set all configuration needed to deploy grid node application in heroku platform.

   .. method:: deploy(self)


      Method to deploy Grid Node app on heroku platform.


   .. method:: __run_heroku_commands(self)


      Add a set of commands/logs used to deploy grid node app on heroku platform.


   .. method:: __check_heroku_dependencies(self)


      Check specific dependencies to perform grid node deploy on heroku platform.



