from formulae.matrices import (
    # DesignMatrices,
    GroupEffectsMatrix,
    CommonEffectsMatrix,
    ResponseMatrix,
)
from formulae import model_description
from formulae.utils import flatten_list
from formulae.environment import Environment
import pandas as pd
import numpy as np

import logging

_log = logging.getLogger("formulae")


class _GroupEffectsMatrix(GroupEffectsMatrix):

    def __init__(self, terms):
        super().__init__(terms)

    def as_dataframe(self):
        """Returns `self.design_matrix` as a pandas.DataFrame."""
        colnames = [term.labels for term in self.terms.values()]
        data = pd.DataFrame(self.design_matrix, columns=list(flatten_list(colnames)))
        mask = np.isclose(data, 0)
        data[mask] = 0
        return data


class DesignMatrices:
    """A wrapper of the response, the common, and group specific effects.

    Parameters
    ----------

    model : Model
        The model description, the result of calling ``model_description``.
    data : pandas.DataFrame
        The data frame where variables are taken from.
    env : Environment
        The environment where values and functions are taken from.

    Attributes
    ----------
    response : ResponseMatrix
        The response in the model. Access its values with ``self.response.design_matrix``. It is
        ``None`` if there is no response term in the model.
    common : CommonEffectsMatrix
        The common effects (a.k.a. fixed effects) in the model. The design matrix can be accessed
        with ``self.common.design_matrix``. The submatrix for a term is accessed via
        ``self.common[term_name]``. It is ``None`` if there are no common terms in the
        model.
    group : GroupEffectsMatrix
        The group specific effects (a.k.a. random effects) in the model. The design matrix can be
        accessed with ``self.group.design_matrix``. The submatrix for a term is accessed via
        ``self.group[term_name]``. It is ``None`` if there are no group specific terms in the
        model.
    """

    def __init__(self, model, data, env):
        self.data = data
        self.env = env
        self.y = None
        self.X = None
        self.common = None
        self.group = None
        self.model = model

        # Evaluate terms in the model
        self.model.eval(data, env)

        if self.model.response:
            self.y = ResponseMatrix(self.model.response)
            self.y.evaluate(data, env)

        if self.model.common_terms:
            self.common = CommonEffectsMatrix(self.model.common_terms)
            self.common.evaluate(data, env)

        if self.model.group_terms:
            self.group = _GroupEffectsMatrix(self.model.group_terms)
            self.group.evaluate(data, env)

        if self.common or self.group:
            self.X = pd.concat(
                [self.common.as_dataframe(), self.group.as_dataframe()], axis=1
            )


def design_matrices(formula, data, na_action="drop", env=0, extra_namespace=None):
    """Parse model formula and obtain a ``DesignMatrices`` object containing objects representing
    the response and the design matrices for both the common and group specific effects.

    Parameters
    ----------
    formula : string
        A model formula.
    data : pandas.DataFrame
        The data frame where variables in the formula are taken from.
    na_action : string
        Describes what to do with missing values in ``data``. ``"drop"`` means to drop
        all rows with a missing value, ``"error"`` means to raise an error,
        ``"pass"`` means to to keep all. Defaults to ``"drop"``.
    env : integer
        The number of environments we walk up in the stack starting from the function's caller
        to capture the environment where formula is evaluated. Defaults to 0 which means
        the evaluation environment is the environment where ``design_matrices`` is called.
    extra_namespace : dict
        Additional user supplied transformations to include in the environment where the formula
        is evaluated. Defaults to ``None``.

    Returns
    -------
    design : DesignMatrices
        An instance of DesignMatrices that contains the design matrice(s) described by
        ``formula``.
    """

    if not isinstance(formula, str):
        raise ValueError("'formula' must be a string.")

    if len(formula) == 0:
        raise ValueError("'formula' cannot be an empty string.")

    if not isinstance(data, pd.DataFrame):
        raise ValueError("'data' must be a pandas.DataFrame.")

    if data.shape[0] == 0:
        raise ValueError("'data' does not contain any observation.")

    if na_action not in ["drop", "error", "pass"]:
        raise ValueError("'na_action' must be either 'drop', 'error' or 'pass'")

    extra_namespace = extra_namespace or {}

    env = Environment.capture(env, reference=1)
    env = env.with_outer_namespace(extra_namespace)

    description = model_description(formula)

    # Incomplete rows are calculated using columns involved in model formula only
    cols_to_select = description.var_names.intersection(set(data.columns))
    data = data[list(cols_to_select)]

    incomplete_rows = data.isna().any(axis=1)
    incomplete_rows_n = incomplete_rows.sum()

    if incomplete_rows_n > 0:
        if na_action == "pass":
            _log.info(
                "Keeping %s/%s rows with at least one missing value in the dataset.",
                incomplete_rows_n,
                data.shape[0],
            )
        elif na_action == "drop":
            _log.info(
                "Automatically removing %s/%s rows from the dataset.",
                incomplete_rows_n,
                data.shape[0],
            )
            data = data[~incomplete_rows]
        else:
            raise ValueError(f"'data' contains {incomplete_rows_n} incomplete rows.")

    design = DesignMatrices(description, data, env)
    return design
