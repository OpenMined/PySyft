:mod:`grid.auth.user_auth`
==========================

.. py:module:: grid.auth.user_auth


Module Contents
---------------

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



