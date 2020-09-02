def threaded(f, daemon=False):
    import queue
    import threading

    def wrapped_f(q, *args, **kwargs):
        """this function calls the decorated function and puts the result in a
        queue."""
        ret = f(*args, **kwargs)
        q.put(ret)

    def wrap(*args, **kwargs):
        """this is the function returned from the decorator.

        It fires off wrapped_f in a new thread and returns the thread
        object with the result queue attached
        """

        q = queue.Queue()

        t = threading.Thread(target=wrapped_f, args=(q,) + args, kwargs=kwargs)
        t.daemon = daemon
        t.start()
        t.result_queue = q
        return t

    return wrap
