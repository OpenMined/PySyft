#!/bin/env python
"""
    Grid Gateway is a Flask based application used to manage / monitor / control  and route  grid workers remotely
"""

from app import create_app
import sys
import os

PORT = os.environ["PORT"]
app = create_app(debug=False)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
