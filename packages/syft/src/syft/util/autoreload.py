AUTORELOAD_ENABLED = False


def enable_autoreload() -> None:
    global AUTORELOAD_ENABLED
    try:
        # third party
        from IPython import get_ipython

        ipython = get_ipython()
        ipython.run_line_magic("load_ext", "autoreload")
        ipython.run_line_magic("autoreload", "2")
        AUTORELOAD_ENABLED = True
    except Exception:
        AUTORELOAD_ENABLED = False


def disable_autoreload() -> None:
    global AUTORELOAD_ENABLED
    try:
        # third party
        from IPython import get_ipython

        ipython = get_ipython()
        ipython.run_line_magic("autoreload", "0")
        AUTORELOAD_ENABLED = False
    except Exception:
        pass


def autoreload_enabled() -> bool:
    global AUTORELOAD_ENABLED
    return AUTORELOAD_ENABLED
