from .roles.blueprint import roles_blueprint
from .users.blueprint import users_blueprint
from .setup.blueprint import setup_blueprint
from .infrastructure.blueprint import infrastructure_blueprint
from .association_requests.blueprint import association_requests_blueprint
from .general.blueprint import root_blueprint


from .users.routes import *
from .roles.routes import *
from .general.routes import *
from .association_requests.routes import *
from .infrastructure.routes import *
from .setup.routes import *
