import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from formulae import design_matrices, model_description
import itertools


def set_datatype(
    df,
    ordinal: list = None,
    categorical: list = None,
    continuous_as_factors: list = None,
    continuous: list = None,
    discrete: list = None,
    inplace=False,
):
    """
    Sets the data types for the DataFrame according to the specified parameters.

    Parameters:
    - ordinal: list or dict, optional
        Specifies the ordinal columns and their order.
    - categorical: list, optional
        Specifies the categorical columns.
    - continuous_as_factors: list, optional
        Specifies the continuous columns to be treated as factors.
    - continuous: list, optional
        Specifies the continuous columns.
    - discrete: list, optional
        Specifies the discrete columns.
    - inplace: bool, default False
        If True, modifies the DataFrame in place.
    """

    if not inplace:
        df = df.copy()

    if ordinal:
        if isinstance(ordinal, dict):
            for col, order in ordinal.items():
                df[col] = pd.Categorical(
                    df[col], categories=order, ordered=True
                ).factorize()[0]
        elif isinstance(ordinal, list):
            if all(isinstance(item, dict) for item in ordinal):
                for ord_dict in ordinal:
                    for col, order in ord_dict.items():
                        df[col] = pd.Categorical(
                            df[col], categories=order, ordered=True
                        ).factorize()[0]
            else:
                for col in ordinal:
                    df[col] = df[col].astype("category").cat.as_ordered().factorize()[0]
        else:
            raise ValueError("ordinal must be a list.")

    if categorical:
        for col in categorical:
            df[col] = df[col].astype("category")

    if continuous_as_factors:
        for col in continuous_as_factors:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if continuous:
        for col in continuous:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if discrete:
        for col in discrete:
            df[col] = df[col].astype(np.int64)

    if inplace:
        return None
    else:
        return df


def make_doe(*args: dict):
    iters = []
    cols = []
    args = [a for a in args if a is not None]
    for factor in args:
        cols.extend(list(factor.keys()))
        iters.extend(factor.values())
    data = list(itertools.product(*iters))
    data = pd.DataFrame(data, columns=cols)
    return data


def generate_doe(*args: dict):
    iters = []
    cols = []
    args = [a for a in args if a is not None]
    for factor in args:
        cols.extend(list(factor.keys()))
        iters.extend(factor.values())
    return cols, list(itertools.product(*iters))


def generate_doe_data(
    ord_data: Optional[Dict[str, Any]] = None,
    cat_data: Optional[Dict[str, Any]] = None,
    nfact_data: Optional[Dict[str, Any]] = None,
    n_cont: Optional[int] = None,
    n_dis: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
):
    if rng is None:
        rng = np.random.default_rng(28)

    cols, data = generate_doe(ord_data, cat_data, nfact_data)
    data = pd.DataFrame(data, columns=cols)

    if n_cont or n_dis:
        N = len(data)
        if n_cont:
            cont_data = rng.normal(size=(N, n_cont))
            num_cols = ["X" + str(i + 1) for i in range(n_cont)]
            data = pd.concat([data, pd.DataFrame(cont_data, columns=num_cols)], axis=1)
        if n_dis:
            dis_data = rng.poisson(lam=10, size=(N, n_dis))
            num_cols = ["Z" + str(i + 1) for i in range(n_dis)]
            data = pd.concat([data, pd.DataFrame(dis_data, columns=num_cols)], axis=1)

    return data
