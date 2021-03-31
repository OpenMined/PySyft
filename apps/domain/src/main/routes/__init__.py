from .roles.blueprint import roles_blueprint
from .users.blueprint import users_blueprint
from .setup.blueprint import setup_blueprint
from .groups.blueprint import groups_blueprint
from .data_centric.blueprint import dcfl_blueprint
from .model_centric.blueprint import mcfl_blueprint
from .association_requests.blueprint import association_requests_blueprint
from .search.blueprint import search_blueprint
from .general.blueprint import root_blueprint


from .users.routes import *
from .general.routes import *
from .groups.routes import *
from .roles.routes import *
from .association_requests.routes import *
from .setup.routes import *
from .data_centric.datasets.routes import *
from .data_centric.requests.routes import *
from .data_centric.workers.routes import *
from .data_centric.tensors.routes import *
from .model_centric.routes import *
from .search.routes import *
