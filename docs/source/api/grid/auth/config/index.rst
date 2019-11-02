:mod:`grid.auth.config`
=======================

.. py:module:: grid.auth.config


Module Contents
---------------

.. function:: register_new_credentials(path: str) -> UserAuthentication

   Create a new credential if not found any credential file during load_credentials function.

   :param path: File path.
   :type path: str

   :returns: New credential instance.
   :rtype: user (UserAuthentication)


.. function:: read_authentication_configs(directory=None, folder=None) -> list

   Search for a path and folder used to store user credentials

   :param directory: System path (can usually be /home/<user>).
   :type directory: str
   :param folder: folder name used to store PyGrid credentials.
   :type folder: str

   :returns: List of credentials instances.
   :rtype: List


.. function:: search_credential(user: str)

   Search for a specific credential instance.

   :param user: Key used to identify the credential.
   :type user: str

   :returns: Credential's instance.
   :rtype: BaseAuthentication


