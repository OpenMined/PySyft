:mod:`grid.auth.authentication`
===============================

.. py:module:: grid.auth.authentication


Module Contents
---------------

.. py:class:: BaseAuthentication(filename)

   Bases: :class:`abc.ABC`

   BaseAuthentication abstract class defines generic methods used by all types of authentications defined in this module.

   .. method:: parse(self)
      :abstractmethod:


      Read, parse and load credential files.


   .. method:: json(self)
      :abstractmethod:


      Convert credential instances into a JSON structure.



