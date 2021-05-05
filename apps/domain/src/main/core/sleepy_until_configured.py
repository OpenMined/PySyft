# stdlib
from json import dumps

# third party
from flask import Response
from werkzeug.wrappers import Request

# grid relative
from ..core.codes import RESPONSE_MSG
from .database import SetupConfig
from .database import db
from .exceptions import AppInSleepyMode


class SleepyUntilConfigured(object):
    def __init__(self, app, app_wsgi):
        self.app = app
        self.wsgi = app_wsgi
        self.allowed_route = {"/setup": ["POST"], "/metadata": ["GET"]}

    @property
    def is_sleeping(self):
        with self.app.app_context():
            return db.session.query(SetupConfig).first() is None

    def is_route_allowed(self, request):
        request_methods = self.allowed_route.get(request.path, [])
        return request.method in request_methods

    def __call__(self, environ, start_response):
        request = Request(environ)
        response_body = {}

        if self.is_sleeping:
            if self.is_route_allowed(request):
                return self.wsgi(environ, start_response)
            else:
                status_code = 401  # Not Allowed
                mimetype = "application/json"
                response_body[RESPONSE_MSG.ERROR] = str(AppInSleepyMode())
                res = Response(
                    dumps(response_body), mimetype=mimetype, status=status_code
                )
                return res(environ, start_response)
        else:
            return self.wsgi(environ, start_response)
