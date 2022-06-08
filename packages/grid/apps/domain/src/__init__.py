# Mono Repo Global Version
# pyproject.toml uses read-version which checks for __version__ assignment
__version__ = "0.5.1"
# elsewhere we can call this file: `python VERSION` and simply take the stdout

if __name__ == "__main__":
    print(__version__)
