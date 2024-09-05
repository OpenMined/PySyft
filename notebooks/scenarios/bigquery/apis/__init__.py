# stdlib
import os

# syft absolute
from syft.util.util import str_to_bool

# relative
from .submit_query import make_submit_query

env_var = "TEST_BIGQUERY_APIS_LIVE"
use_live = str_to_bool(str(os.environ.get(env_var, "False")))
env_name = "Live" if use_live else "Mock"
print(f"Using {env_name} API Code, this will query BigQuery. ${env_var}=={use_live}")


if use_live:
    # relative
    from .live.schema import make_schema
    from .live.test_query import make_test_query
else:
    # relative
    from .simulation.schema import make_schema
    from .simulation.test_query import make_test_query
