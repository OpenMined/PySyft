from utils import User, body_to_obj
from syft_rpc import Request

user = User(id=1, name="Alice")

request = Request()

headers = {}
headers["content-type"] = "application/json"
headers["x-syft-rpc-object-type"] = type(user).__name__
response = request.get(
    "syft://madhava@openmined.org/public/rpc/test/listen",
    body=user.dump(),
    headers=headers,
)

result = response.wait()

result = body_to_obj(response)
print(result)
