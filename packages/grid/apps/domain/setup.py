# -*- coding: utf-8 -*-
from setuptools import setup

package_dir = {"": "src"}

modules = ["domain"]
install_requires = [
    "Flask-Cors>=3.0.10,<4.0.0",
    "Flask-Executor>=0.9.4,<0.10.0",
    "Flask-Sockets>=0.2.1,<0.3.0",
    "Flask>=1.1.1,<2.0.0",
    # "PyInquirer>=1.0.3,<2.0.0",
    "pynacl",
    "PyJWT>=1.7.1,<2.0.0",
    "bcrypt>=3.2.0,<4.0.0",
    "boto3>=1.14.51,<2.0.0",
    "flask_migrate>=2.5.3,<3.0.0",
    "flask_sqlalchemy>=2.4.4,<3.0.0",
    "gevent-websocket>=0.10.1,<0.11.0",
    "gevent>=20.6.2,<21.0.0",
    "gunicorn>=20.0.4,<21.0.0",
    "loguru>=0.5.3,<0.6.0",
    "numpy>=1.18.5,<2.0.0",
    "requests-toolbelt==0.9.1",
    "scipy>=1.6.1,<2.0.0",
    "sqlalchemy_mixins>=1.2.1,<2.0.0",
    "sqlalchemy_utils>=0.36.8,<0.37.0",
    "sqlitedict>=1.6.0,<2.0.0",
    "tenseal>=0.3.2,<0.4.0",
    "terrascript>=0.9.0,<0.10.0",
    "textwrap3>=0.9.2,<0.10.0",
]

setup_kwargs = {
    "name": "domain",
    "version": "0.3.0",
    "description": "",
    "long_description": None,
    "author": "Ionesio Junior",
    "author_email": "ionesiojr@gmail.com",
    "maintainer": None,
    "maintainer_email": None,
    "url": None,
    "package_dir": package_dir,
    "py_modules": modules,
    "install_requires": install_requires,
    "python_requires": ">=3.8,<4.0",
}


setup(**setup_kwargs)
