from http.server import HTTPServer, BaseHTTPRequestHandler

class CustomHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('content-type','text/html')
        self.end_headers()
        self.wfile.write('Network Node running.'.encode())

def main():
    srv = HTTPServer(('',80), CustomHandler)
    srv.serve_forever()

if __name__ == '__main__':
    main()
