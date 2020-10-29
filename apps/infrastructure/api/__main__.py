import os
import json
from pathlib import Path
from flask import Flask, Response, jsonify, request

from .providers.aws import AWS_Serverfull, AWS_Serverless

app = Flask(__name__)


@app.route("/", methods=["POST"])
def index():
    """
    Deploys the resources.
    """

    data = json.loads(request.json)

    provider = data.get("provider").lower()
    deployment_type = data.get("deployment_type").lower()

    if provider == "aws":
        if deployment_type == "serverless":
            aws_deployment = AWS_Serverless(
                credentials=data["credentials"]["cloud"],
                vpc_config=data["vpc"],
                db_config=data["credentials"]["db"],
                app_config=data["app"],
            )
            aws_deployment.deploy()
        elif deployment_type == "serverfull":
            pass
    elif provider == "azure":
        pass
    elif provider == "gcp":
        pass

    response = {"message": "Deployment successful"}

    return Response(json.dumps(response), status=200, mimetype="application/json")


if __name__ == "__main__":
    app.run()
