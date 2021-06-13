# stdlib
import functools
from typing import Any as TypeAny
from typing import List as TypeList
from typing import Tuple as TypeTuple

# third party
import statsmodels
import statsmodels.api as sm

# syft relative
from .serde import family  # noqa: 401
from .serde import results  # noqa: 401
from syft.ast import add_classes
from syft.ast import add_methods
from syft.ast import add_modules
from syft.ast.globals import Globals
from syft.lib.util import generic_update_ast

# import wrappers
from .serde.results import wrap_me as wrap_results_kwargs
from .serde.family import wrap_me_list as wrap_family_list_kwargs

LIB_NAME = "statsmodels"
PACKAGE_SUPPORT = {"lib": LIB_NAME}


def create_ast(client: TypeAny = None) -> Globals:
    ast = Globals(client=client)

    modules: TypeList[TypeTuple[str, TypeAny]] = [
        ("statsmodels", statsmodels),
        ("statsmodels.api", sm),
        ("statsmodels.genmod", statsmodels.genmod),
        (
            "statsmodels.genmod.generalized_linear_model",
            statsmodels.genmod.generalized_linear_model,
        ),
        ("statsmodels.genmod.families", statsmodels.genmod.families),
        ("statsmodels.iolib", statsmodels.iolib),
        ("statsmodels.iolib.summary", statsmodels.iolib.summary),
    ]

    classes: TypeList[TypeTuple[str, str, TypeAny]] = [
        (
            "statsmodels.genmod.generalized_linear_model.GLM",
            "statsmodels.genmod.generalized_linear_model.GLM",
            statsmodels.genmod.generalized_linear_model.GLM,
        ),
        (
            "statsmodels.genmod.generalized_linear_model.GLMResults",
            "statsmodels.genmod.generalized_linear_model.GLMResults",
            statsmodels.genmod.generalized_linear_model.GLMResults,
        ),
        (
            "statsmodels.genmod.generalized_linear_model.GLMResultsWrapper",
            "statsmodels.genmod.generalized_linear_model.GLMResultsWrapper",
            statsmodels.genmod.generalized_linear_model.GLMResultsWrapper,
        ),
        (
            "statsmodels.iolib.summary.Summary",
            "statsmodels.iolib.summary.Summary",
            statsmodels.iolib.summary.Summary,
        ),
        (
            "statsmodels.genmod.families.Binomial",
            "statsmodels.genmod.families.Binomial",
            statsmodels.genmod.families.Binomial,
        ),
        (
            "statsmodels.genmod.families.Gamma",
            "statsmodels.genmod.families.Gamma",
            statsmodels.genmod.families.Gamma,
        ),
        (
            "statsmodels.genmod.families.Gaussian",
            "statsmodels.genmod.families.Gaussian",
            statsmodels.genmod.families.Gaussian,
        ),
        (
            "statsmodels.genmod.families.InverseGaussian",
            "statsmodels.genmod.families.InverseGaussian",
            statsmodels.genmod.families.InverseGaussian,
        ),
        (
            "statsmodels.genmod.families.NegativeBinomial",
            "statsmodels.genmod.families.NegativeBinomial",
            statsmodels.genmod.families.NegativeBinomial,
        ),
        (
            "statsmodels.genmod.families.Poisson",
            "statsmodels.genmod.families.Poisson",
            statsmodels.genmod.families.Poisson,
        ),
        (
            "statsmodels.genmod.families.Tweedie",
            "statsmodels.genmod.families.Tweedie",
            statsmodels.genmod.families.Tweedie,
        ),
    ]

    # TODO: finish all these methods, summary is an object attribute for example
    methods = [
        ("statsmodels.api.add_constant", "pandas.DataFrame"),
        (
            "statsmodels.genmod.generalized_linear_model.GLM.fit",
            "statsmodels.genmod.generalized_linear_model.GLMResultsWrapper",
        ),
        (
            "statsmodels.genmod.generalized_linear_model.GLMResults.summary",
            "statsmodels.iolib.summary.Summary",
        ),
        (
            "statsmodels.genmod.generalized_linear_model.GLMResultsWrapper.cov_params",
            "pandas.DataFrame",
        ),
        (
            "statsmodels.genmod.generalized_linear_model.GLMResultsWrapper.conf_int",
            "pandas.DataFrame",
        ),
        ("statsmodels.iolib.summary.Summary.as_csv", "syft.lib.python.String"),
        ("statsmodels.iolib.summary.Summary.as_html", "syft.lib.python.String"),
        ("statsmodels.iolib.summary.Summary.as_latex", "syft.lib.python.String"),
        ("statsmodels.iolib.summary.Summary.as_text", "syft.lib.python.String"),
    ]

    add_modules(ast, modules)
    add_classes(ast, classes)
    add_methods(ast, methods)

    for klass in ast.classes:
        klass.create_pointer_class()
        klass.create_send_method()
        klass.create_storable_object_attr_convenience_methods()

    return ast


update_ast = functools.partial(generic_update_ast, LIB_NAME, create_ast)

objects = family.wrap_me_list
objects.append(results.wrap_me)
