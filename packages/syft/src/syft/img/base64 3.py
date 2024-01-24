# stdlib
import base64


# allows images to work with offline mode
def base64read(fname):
    # relative
    from .. import SYFT_PATH

    with open(SYFT_PATH / "img" / fname, "rb") as file:
        res = base64.b64encode(file.read())
        return f"data:image/png;base64,{res.decode('utf-8')}"
