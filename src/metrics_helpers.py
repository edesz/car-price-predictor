#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import List, Union

import sklearn.metrics as mr
from numpy import abs, array, mean
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator

iterables = Union[array, List, Series]


def mean_absolute_percentage_error(
    y_true: Union[array, List, Series], y_pred: Union[array, List, Series]
) -> float:
    """
    Calculate MAPE
    Inputs
    ------
    y_true : list, np.array or pd.Series
        true observations
    y_pred : list, np.array or pd.Series
        predictions
    """
    return mean(abs((y_true - y_pred) / y_true)) * 100


def mae(r: iterables, f: iterables) -> float:
    """
    Calculate Maximum Error
    Inputs
    ------
    r : list, np.array or pd.Series
        true observations
    f : list, np.array or pd.Series
        predictions
    """
    return mr.max_error(r, f)


def maxe(r: iterables, f: iterables) -> float:
    """
    Calculate MAE
    Inputs
    ------
    r : list, np.array or pd.Series
        true observations
    f : list, np.array or pd.Series
        predictions
    """
    return mr.mean_absolute_error(r, f)


def mdae(r: iterables, f: iterables) -> float:
    """
    Calculate Median Absolute Error
    Inputs
    ------
    r : list, np.array or pd.Series
        true observations
    f : list, np.array or pd.Series
        predictions
    """
    return mr.median_absolute_error(r, f)


def mse(r: iterables, f: iterables) -> float:
    """
    Calculate MSE
    Inputs
    ------
    r : list, np.array or pd.Series
        true observations
    f : list, np.array or pd.Series
        predictions
    """
    return mr.mean_squared_error(r, f)


def rmse(r: iterables, f: iterables) -> float:
    """
    Calculate RMSE
    Inputs
    ------
    r : list, np.array or pd.Series
        true observations
    f : list, np.array or pd.Series
        predictions
    """
    return mr.mean_squared_error(r, f, squared=False)


def r2score(r: iterables, f: iterables) -> float:
    """
    Calculate R^2
    Inputs
    ------
    r : list, np.array or pd.Series
        true observations
    f : list, np.array or pd.Series
        predictions
    """
    return mr.r2_score(r, f)


def get_test_metrics(
    metrics_wanted: List,
    X_r: DataFrame,
    r: iterables,
    f: iterables,
    est: BaseEstimator,
) -> DataFrame:
    df_metrics = DataFrame.from_dict(
        {
            "MAE": mae(r, f),
            "MDAE": mdae(r, f),
            "RMSE": rmse(r, f),
            "MAXE": maxe(r, f),
            "MAPE (%)": mean_absolute_percentage_error(r, f),
            "R2": r2score(r, f),
            ".score()": est.score(X_r, r),
        },
        orient="index",
        columns=["value"],
    )
    return df_metrics
