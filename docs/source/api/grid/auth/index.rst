:mod:`grid.auth`
================

.. py:module:: grid.auth


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   authentication/index.rst
   config/index.rst
   user_auth/index.rst


Package Contents
----------------

.. py:class:: UserAuthentication(username, password)

   Bases: :class:`grid.auth.authentication.BaseAuthentication`

   .. attribute:: FILENAME
      :annotation: = auth.user

      

   .. attribute:: USERNAME_FIELD
      :annotation: = username

      

   .. attribute:: PASSWORD_FIELD
      :annotation: = password

      

   .. method:: parse(path)
      :staticmethod:


      Static method used to create new user authentication instances parsing a json file.

      :param path: json file path.
      :type path: str

      :returns: List of user authentication objects.
      :rtype: List


   .. method:: json(self)


      Convert user instances into a JSON/Dictionary structure.



.. data:: BASE_DIR
   

   

.. data:: BASE_FOLDER
   :annotation: = .openmined

   

.. data:: AUTH_MODELS
   

   

.. data:: auth_credentials
   :annotation: = []

   

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


