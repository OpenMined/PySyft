AUTORELOAD_ENABLED = False


def enable_autoreload() -> None:
    global AUTORELOAD_ENABLED
    try:
        # third party
        from IPython import get_ipython

        ipython = get_ipython()  # noqa: F821
        if hasattr(ipython, "run_line_magic"):
            ipython.run_line_magic("load_ext", "autoreload")
            ipython.run_line_magic("autoreload", "2")
        AUTORELOAD_ENABLED = True
        print("Autoreload enabled")
    except Exception as e:
        AUTORELOAD_ENABLED = False
        print(f"Error enabling autoreload: {e}")


def disable_autoreload() -> None:
    global AUTORELOAD_ENABLED
    try:
        # third party
        from IPython import get_ipython

        ipython = get_ipython()  # noqa: F821
        if hasattr(ipython, "run_line_magic"):
            ipython.run_line_magic("autoreload", "0")
        AUTORELOAD_ENABLED = False
        print("Autoreload disabled.")
    except Exception as e:
        print(f"Error disabling autoreload: {e}")


def autoreload_enabled() -> bool:
    global AUTORELOAD_ENABLED
    return AUTORELOAD_ENABLED
