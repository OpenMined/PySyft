import json
import os
from pathlib import Path

from flask import Flask, Response, jsonify, request
from loguru import logger

from .providers.aws import AWS_Serverfull, AWS_Serverless
from .utils import Config

app = Flask(__name__)


@app.route("/")
def index():
    response = {
        "message": "Welcome to OpenMined PyGrid Infrastructure Deployment Suite"
    }
    return Response(json.dumps(response), status=200, mimetype="application/json")


@app.route("/deploy", methods=["POST"])
def deploy():
    """Deploys the resources."""

    data = json.loads(request.json)
    config = Config(**data)

    deployed = False
    output = None

    if config.provider == "aws":
        if config.serverless:
            aws_deployment = AWS_Serverless(config)
            deployed, output = aws_deployment.deploy()
        else:
            aws_deployment = AWS_Serverfull(config=config)
            deployed, output = aws_deployment.deploy()
    elif config.provider == "azure":
        pass
    elif config.provider == "gcp":
        pass

    response = {
        "message": f"Your PyGrid {config.app.name} was deployed successfully"
        if deployed
        else f"Your attempt to deploy PyGrid {config.app.name} failed",
        "output": output,
    }
    return Response(json.dumps(response), status=200, mimetype="application/json")
