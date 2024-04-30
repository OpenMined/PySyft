# stdlib
import doctest
import os
from unittest import TestCase


class ReadMeDocTest(TestCase):
    def test__readme_doctests(self):
        readme_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "README.rst")
        )
        self.assertTrue(os.path.exists(readme_path))
        result = doctest.testfile(readme_path, module_relative=False)
        self.assertEqual(result.failed, 0, "%s tests failed!" % result.failed)
