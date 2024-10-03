# stdlib
import os

# third party
from ipywidgets.widgets import Output as ipyOutput


class Output(ipyOutput):
    # This is a workaround for the issue that
    # the Output widget causes halt when running in Jupyter Notebook
    # from cli, e.g. tests.
    #
    # No-op when running in Jupyter Notebook.
    if "JPY_PARENT_PID" in os.environ:

        def __enter__(self):  # type: ignore
            pass

        def __exit__(self, *args, **kwargs):  # type: ignore
            pass
