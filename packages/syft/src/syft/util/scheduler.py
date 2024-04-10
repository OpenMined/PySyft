# stdlib
import threading


class TaskScheduler:
    def __init__(self):
        self.tasks = []
        self.lock = threading.Lock()

    def add_task(self, task):
        with self.lock:
            self.tasks.append(task)

    def start(self):
        for task in self.tasks:
            thread = threading.Thread(target=task)
            thread.start()

        # Wait for all threads to finish
        for thread in threading.enumerate():
            if thread != threading.current_thread():
                thread.join()

    def clear(self):
        with self.lock:
            self.tasks.clear()
