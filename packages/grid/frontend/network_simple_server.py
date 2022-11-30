# stdlib
from http.server import BaseHTTPRequestHandler
from http.server import HTTPServer


class CustomHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("content-type", "text/html")
        self.end_headers()
        self.wfile.write("Network Node running.".encode())


def main() -> None:
    srv = HTTPServer(("", 80), CustomHandler)
    srv.serve_forever()


if __name__ == "__main__":
    main()
