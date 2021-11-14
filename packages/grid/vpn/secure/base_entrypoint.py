# type: ignore
# system imports
# stdlib
from collections import OrderedDict
from typing import Any
from typing import Callable
from typing import Dict
from typing import List

# third party
# web imports
from flask_executor import Executor
from flask_executor.futures import Future
from flask_shell2http.helpers import get_logger

# relative
# lib imports
from .api import Shell2HttpAPI

logger = get_logger()


class Shell2HTTP(object):
    """
    Flask-Shell2HTTP base entrypoint class.
    The only public API available to users.

    Attributes:
        app: Flask application instance.
        executor: Flask-Executor instance
        base_url_prefix (str): base prefix to apply to endpoints. Defaults to "/".

    Example::

        app = Flask(__name__)
        executor = Executor(app)
        shell2http = Shell2HTTP(app=app, executor=executor, base_url_prefix="/tasks/")
    """

    __commands: "OrderedDict[str, str]" = OrderedDict()
    __url_prefix: str = "/"

    def __init__(
        self, app=None, executor: Executor = None, base_url_prefix: str = "/"
    ) -> None:
        self.__url_prefix = base_url_prefix
        if app and executor:
            self.init_app(app, executor)

    def init_app(self, app, executor: Executor) -> None:
        """
        For use with Flask's `Application Factory`_ method.

        Example::

            executor = Executor()
            shell2http = Shell2HTTP(base_url_prefix="/commands/")
            app = Flask(__name__)
            executor.init_app(app)
            shell2http.init_app(app=app, executor=executor)

        .. _Application Factory:
           https://flask.palletsprojects.com/en/1.1.x/patterns/appfactories/
        """
        self.app = app
        self.__executor: Executor = executor
        self.__init_extension()

    def __init_extension(self) -> None:
        """
        Adds the Shell2HTTP() instance to `app.extensions` list.
        For internal use only.
        """
        if not hasattr(self.app, "extensions"):
            self.app.extensions = dict()

        self.app.extensions["shell2http"] = self

    def register_command(
        self,
        endpoint: str,
        command_name: str,
        callback_fn: Callable[[Dict, Future], Any] = None,
        decorators: List = [],
    ) -> None:
        """
        Function to map a shell command to an endpoint.

        Args:
            endpoint (str):
                - your command would live here: ``/{base_url_prefix}/{endpoint}``
            command_name (str):
                - The base command which can be executed from the given endpoint.
                - If ``command_name='echo'``, then all arguments passed
                  to this endpoint will be appended to ``echo``.\n
                  For example,
                  if you pass ``{ "args": ["Hello", "World"] }``
                  in POST request, it gets converted to ``echo Hello World``.\n
            callback_fn (Callable[[Dict, Future], Any]):
                - An optional function that is invoked when a requested process
                    to this endpoint completes execution.
                - This is added as a
                    ``concurrent.Future.add_done_callback(fn=callback_fn)``
                - The same callback function may be used for multiple commands.
                - if request JSON contains a `callback_context` attr, it will be passed
                  as the first argument to this function.
            decorators (List[Callable]):
                - A List of view decorators to apply to the endpoint.
                - *New in version v1.5.0*

        Examples::

            def my_callback_fn(context: dict, future: Future) -> None:
                print(future.result(), context)

            shell2http.register_command(endpoint="echo", command_name="echo")
            shell2http.register_command(
                endpoint="myawesomescript",
                command_name="./fuxsocy.py",
                callback_fn=my_callback_fn,
                decorators=[],
            )
        """
        uri: str = self.__construct_route(endpoint)
        # make sure the given endpoint is not already registered
        cmd_already_exists = self.__commands.get(uri)
        if cmd_already_exists:
            logger.error(
                "Failed to register since given endpoint: "
                f"'{endpoint}' already maps to command: '{cmd_already_exists}'."
            )
            return None

        # else, add new URL rule
        view_func = Shell2HttpAPI.as_view(
            endpoint,
            command_name=command_name,
            user_callback_fn=callback_fn,
            executor=self.__executor,
        )
        # apply decorators, if any
        for dec in decorators:
            view_func = dec(view_func)
        # register URL rule
        self.app.add_url_rule(
            uri,
            view_func=view_func,
        )
        self.__commands.update({uri: command_name})
        logger.info(f"New endpoint: '{uri}' registered for command: '{command_name}'.")

    def get_registered_commands(self) -> "OrderedDict[str, str]":
        """
        Most of the time you won't need this since
        Flask provides a ``Flask.url_map`` attribute.

        Returns:
            OrderedDict[uri, command]
            i.e. mapping of registered commands and their URLs.
        """
        return self.__commands

    def __construct_route(self, endpoint: str) -> str:
        """
        For internal use only.
        """
        return self.__url_prefix + endpoint
