# stdlib
import builtins
import inspect
import os
import threading

original_open = builtins.open

# Global dictionary for tracking open file information per process and thread
open_file_info = {}


def update_file_info(process_id, thread_id, file_path, increment=True):
    if process_id not in open_file_info:
        open_file_info[process_id] = {}

    if thread_id not in open_file_info[process_id]:
        open_file_info[process_id][thread_id] = {}

    if increment:
        if file_path not in open_file_info[process_id][thread_id]:
            open_file_info[process_id][thread_id][file_path] = 0
        open_file_info[process_id][thread_id][file_path] += 1
    else:
        if open_file_info[process_id][thread_id].get(file_path, 0) > 0:
            open_file_info[process_id][thread_id][file_path] -= 1

    print_current_file_info()


def print_current_file_info():
    for process_id, threads in open_file_info.items():
        for thread_id, files in threads.items():
            for file_path, count in files.items():
                if count > 0:
                    print(
                        f"Process {process_id}, Thread {thread_id}, File {file_path}: {count} open instances"
                    )


class FileWrapper:
    def __init__(self, file_obj, opened_at, file_path):
        self.file_obj = file_obj
        self.process_id = os.getpid()
        self.thread_id = threading.get_ident()
        self.opened_at = opened_at
        self.file_path = file_path

        update_file_info(
            self.process_id, self.thread_id, self.file_path, increment=True
        )

    def __iter__(self):
        return iter(self.file_obj)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        update_file_info(
            self.process_id, self.thread_id, self.file_path, increment=False
        )
        self.file_obj.close()

    def __getattr__(self, item):
        return getattr(self.file_obj, item)


def patched_open(*args, **kwargs):
    file_path = args[0] if args else kwargs.get("file", "")
    try:
        file_obj = original_open(*args, **kwargs)
    except OSError as e:
        if e.errno == 24:  # Too many open files
            print("Exception: Too many open files. Current open files:")
            print_current_file_info()
        raise
    return FileWrapper(
        file_obj,
        f"{inspect.stack()[1].filename}:{inspect.stack()[1].lineno}",
        file_path,
    )


builtins.open = patched_open
