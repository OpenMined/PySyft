:mod:`grid.deploy.heroku_gateway`
=================================

.. py:module:: grid.deploy.heroku_gateway


Module Contents
---------------

.. py:class:: HerokuGatewayDeployment(app_name: str, verbose=True, check_deps=True, dev_user: str = 'OpenMined', branch: set = 'dev', env_vars={})

   Bases: :class:`grid.deploy.BaseDeployment`

   An abstraction of heroku grid gateway deployment process, the purpose of this class is set all configuration needed to deploy grid gateway application in heroku platform.

   .. method:: deploy(self)


      Method to deploy Grid Gateway app on heroku platform.


   .. method:: __run_heroku_commands(self)


      Add a set of commands/logs used to deploy grid gateway app on heroku platform.


   .. method:: __check_heroku_dependencies(self)


      Check specific dependencies to perform grid gateway deploy on heroku platform.



