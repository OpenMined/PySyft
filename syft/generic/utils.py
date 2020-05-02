class memorize(dict):
    """
    This is a decorator to cache a function output when the function is
    deterministic and the input space is small. In such condition, the
    function will be called many times to perform the same computation
    so we want this computation to be cached.
    """

    def __init__(self, func):
        self.func = func

    def __call__(self, *args):
        return self[args]

    def __missing__(self, key):
        result = self[key] = self.func(*key)
        return result
