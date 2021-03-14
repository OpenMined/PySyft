import json
import os
import time
from datetime import datetime
from pathlib import Path

from flask import Flask, Response, jsonify, request
from loguru import logger

from apps.infrastructure.providers import AWS_Serverfull, AWS_Serverless
from apps.infrastructure.providers.provider import Provider
from apps.infrastructure.utils import Config

from .models import Domain, db

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///pygrid.db"
db.init_app(app)

states = {"creating": 0, "failed": 1, "success": 2, "destroyed": 3, "autoscaled": 4}


@app.route("/")
def index():
    response = {
        "message": "Welcome to OpenMined PyGrid Infrastructure Deployment Suite"
    }
    return Response(json.dumps(response), status=200, mimetype="application/json")


@app.route("/deploy", methods=["POST"])
def deploy():
    """Deploys the resources."""
    db.create_all()

    data = json.loads(request.data.decode("utf-8"))
    config = Config(**data)

    deployment = None
    deployed = False
    output = {}

    config.app.id = db.session.query(Domain).count() + 1

    if config.provider == "aws":
        deployment = (
            AWS_Serverless(config)
            if config.serverless
            else AWS_Serverfull(config=config)
        )
    elif config.provider == "azure":
        pass
    elif config.provider == "gcp":
        pass

    if deployment.validate():
        domain = Domain(
            id=config.app.id,
            provider=config.provider,
            region=config.vpc.region,
            instance_type=config.vpc.instance_type.InstanceType,
            state=states["creating"],
        )
        db.session.add(domain)
        db.session.commit()

        deployed, output = deployment.deploy()

        domain = Domain.query.get(config.app.id)
        if deployed:
            domain.state = states["success"]
            domain.deployed_at = datetime.now()
        else:
            domain.state = states["failed"]
        db.session.commit()
    else:
        deployed, output = (
            False,
            {"failure": f"Your attempt to deploy PyGrid {config.app.name} failed"},
        )

    response = {"deloyed": deployed, "output": output}
    return Response(json.dumps(response), status=200, mimetype="application/json")


@app.route("/domains", methods=["GET"])
def get_domains():
    """Get all deployed domains.
    Only Node operators can access this endpoint.
    """
    domains = Domain.query.order_by(Domain.created_at).all()
    return Response(
        json.dumps([domain.as_dict() for domain in domains], default=str),
        status=200,
        mimetype="applications/json",
    )


@app.route("/domains/<int:id>", methods=["GET"])
def get_domain(id):
    """Get specific domain data.
    Only the Node owner and the user who created this worker can access this endpoint.
    """
    domain = Domain.query.get(id)
    return Response(
        json.dumps(domain.as_dict(), default=str),
        status=200,
        mimetype="applications/json",
    )
