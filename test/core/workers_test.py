from unittest import TestCase
import os
import sys
import time
from threading import Thread
from multiprocessing import Process
import syft as sy
from syft.core.workers import WebSocketWorker


class TestBaseWorker(TestCase):
    def test_search_obj(self):

        hook = sy.TorchHook()

        hook.local_worker.is_client_worker = False

        sy.Var(sy.FloatTensor([-2, -1, 0, 1, 2, 3])).set_id(
            "#boston_housing #target #dataset"
        )
        sy.Var(sy.FloatTensor([-2, -1, 0, 1, 2, 3])).set_id(
            "#boston_housing #input #dataset"
        )

        hook.local_worker.is_client_worker = True

        assert len(hook.local_worker.search("#boston_housing")) == 2
        assert len(hook.local_worker.search(["#boston_housing", "#target"])) == 1


class TestWebSocketWorker(TestCase):

    def test_sending(self):

        # We redirect server's stdout and stderr to a pipe
        # to let the client know when the server is ready.
        server_pipe_output_fd, server_pipe_input_fd = os.pipe()
        server_pipe_output = os.fdopen(server_pipe_output_fd, 'r')
        server_pipe_input = os.fdopen(server_pipe_input_fd, 'w')

        def __server():
            os.close(server_pipe_output_fd)
            sys.stdout = sys.stderr = server_pipe_input

            hook = sy.TorchHook()
            WebSocketWorker(hook=hook,
                            id=2,
                            port=8181,
                            is_pointer=False,
                            is_client_worker=False,
                            verbose=True)

        server_process = Process(target=__server, args=())
        server_process.start()

        server_status = "started"
        #  Note:
        #  Later server_status is modified in different threads, but it's ok:
        #  1. Simple assignments are thread-safe.
        #  2. With the current flow of execution the thread which makes
        #     the second modification (server_status = "stopped") waits
        #     for the first modification (while server_status != "initialized")

        os.close(server_pipe_input_fd)

        def __readln_print_server_msg_loop():
            nonlocal server_status

            while server_status != "stopped":
                server_msg = server_pipe_output.readline().strip()
                if server_msg:
                    print("Server: ", server_msg)
                if server_msg == WebSocketWorker.SERVER_INITIALIZED_MSG:
                    server_status = "initialized"

        server_output_print_thread = Thread(target=__readln_print_server_msg_loop)
        server_output_print_thread.start()


        print("Waiting for server initialization.")
        while server_status != "initialized":
            time.sleep(0.1)


        print("Client starts...")
        hook = sy.TorchHook(local_worker=WebSocketWorker(id=111, port=8182))
        remote_client = WebSocketWorker(hook=hook, id=2, port=8181,
                                        is_pointer=True)
        hook.local_worker.add_worker(remote_client)
        x = sy.torch.FloatTensor([1, 2, 3, 4, 5]).send(remote_client)
        x2 = sy.torch.FloatTensor([1, 2, 3, 4, 4]).send(remote_client)
        y = x + x2 + x

        assert y.get() == (3, 6, 9, 12, 14)


        # Cleaning
        server_process.terminate()
        server_process.join()
        server_status = "stopped"
        server_output_print_thread.join()
        os.close(server_pipe_output_fd)
