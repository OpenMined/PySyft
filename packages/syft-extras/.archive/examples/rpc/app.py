from syft_event import Response, Server
import time
from utils import LoginResponse, body_to_obj
from syftbox.lib import Client

client = Client.load()
print("> Client", client.email)
app = Server(app_name="test", client=client, message_timeout=120)


@app.get("/public/rpc/test/listen")
def login(request):
    print("Request Headers", request.headers)
    print("Request Body", request.decode())

    user = body_to_obj(request)

    result = LoginResponse(username=user.name, token=1)
    headers = {}

    headers["content-type"] = "application/json"
    headers["x-syft-rpc-object-type"] = type(result).__name__

    time.sleep(10)

    return Response(content=result, status_code=200, headers=headers)


if __name__ == "__main__":
    app.run()
