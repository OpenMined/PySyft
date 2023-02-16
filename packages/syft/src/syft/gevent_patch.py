# stdlib
import os
from typing import Optional


def str_to_bool(bool_str: Optional[str]) -> bool:
    result = False
    bool_str = str(bool_str).lower()
    if bool_str == "true" or bool_str == "1":
        result = True
    return result


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


GEVENT_MONKEYPATCH = str_to_bool(os.environ.get("GEVENT_MONKEYPATCH", "False"))
jupyter_notebook = is_notebook()

if GEVENT_MONKEYPATCH or jupyter_notebook:
    # third party
    from gevent import monkey

    # 🟡 TODO 30: Move this to where we manage the different concurrency modes later
    # make sure its stable in containers and other run targets
    thread = not jupyter_notebook
    monkey.patch_all(ssl=False, thread=thread)
