#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DFDropDupicates(TransformerMixin):
    def __init__(self, keep="first"):
        self.keep = keep

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop_duplicates(keep=self.keep)


class DFFilterNumerical(TransformerMixin):
    def __init__(
        self, col="per_month_min", numerical_value=0, filter_type="lt"
    ):
        self.numerical_value = numerical_value
        self.col = col
        self.filter_type = filter_type

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.filter_type == "lt":
            return X[X[self.col] < self.numerical_value]
        elif self.filter_type == "ne":
            return X[X[self.col] != self.numerical_value]
        elif "vc" in self.filter_type:
            vc = X[self.col].value_counts()
            if self.filter_type == "vc_lt":
                mask = vc[vc < self.numerical_value]
            elif self.filter_type == "vc_isin":
                mask = vc[vc.index.isin(self.numerical_value)]
            keys = mask.index.tolist()
            X = X[~X[self.col].isin(keys)]
            return X


class DFPolynomialFeatureGenerator(TransformerMixin):
    def __init__(self, col, polynomial_degree=0.5):
        self.polynomial_degree = polynomial_degree
        self.col = col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        deg = (
            f"{str(self.polynomial_degree).replace('.', '')}"
            if type(self.polynomial_degree) == float
            else str(self.polynomial_degree)
        )
        X[f"{self.col}{deg}"] = X[self.col] ** self.polynomial_degree
        return X


class DFInteractionTerms(TransformerMixin):
    def __init__(self, f1="f1", f2="f2"):
        self.f1 = f1
        self.f2 = f2

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X[f"{self.f1}_x_{self.f2}"] = X[self.f1] * X[self.f2]
        return X


class DFDropNa(TransformerMixin):
    def __init__(self, cols=["col1", "col2"], how="any"):
        self.how = how
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        Xs = X.dropna(subset=self.cols, how=self.how)
        return Xs


class DFColTypeChanger(TransformerMixin):
    def __init__(self, cols, new_dtype="str"):
        self.new_dtype = new_dtype
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X[self.cols] = X[self.cols].astype(self.new_dtype)
        return X


class DFOneHotEncoder(TransformerMixin):
    def __init__(self, cols, prefix_sep="="):
        self.prefix_sep = prefix_sep
        self.cols = cols

    def fit(self, X, y=None):
        self.ohe = OneHotEncoder(handle_unknown="ignore")
        self.ohe.fit(X[self.cols])
        return self

    def transform(self, X):
        X_dummies = pd.DataFrame(
            self.ohe.transform(X[self.cols]).toarray(),
            columns=self.ohe.get_feature_names(self.cols),
            index=X.index,
        )
        X_dummies.columns = [
            c[::-1].replace("_", "=", 1)[::-1] for c in list(X_dummies)
        ]

        X_nums = X.drop(self.cols, axis=1)
        X_new = pd.concat([X_nums, X_dummies], axis=1)

        self.column_names = list(X_nums) + X_dummies.columns.tolist()
        return X_new

    def get_feature_names(self):
        return self.column_names


class DFStandardScaler(TransformerMixin):
    # StandardScaler but for pandas DataFrames

    def __init__(self):
        self.ss = None
        self.mean_ = None
        self.scale_ = None

    def fit(self, X, y=None):
        self.ss = StandardScaler()
        self.ss.fit(X)
        self.mean_ = pd.Series(self.ss.mean_, index=X.columns)
        self.scale_ = pd.Series(self.ss.scale_, index=X.columns)
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xss = self.ss.transform(X)
        Xscaled = pd.DataFrame(Xss, index=X.index, columns=X.columns)
        return Xscaled


def generate_dummies(df, dummy_cols, dummy_prefix):
    """One hot encoding for a single column"""
    makes_dummies = pd.get_dummies(
        df[dummy_cols], drop_first=True, prefix=dummy_prefix
    )
    dummy_cols_new = list(makes_dummies)
    # print(dummy_cols_new)
    df.drop(dummy_cols, axis=1, inplace=True)
    df = df.join(makes_dummies)
    return df, dummy_cols_new


def filter_by_value_counts(df, colname, mask):
    """Filter DataFrame using mask"""
    cats_keys = mask.index.tolist()
    df = df[~df[colname].isin(cats_keys)]
    return df


def get_model_coefs(coefs_array, model_name, selected_cols):
    """Extract model coefficients into DataFrame"""
    df_coefs = pd.DataFrame(coefs_array).T
    # df_coefs.index.name = model_name
    df_coefs.columns = selected_cols
    df_coefs = df_coefs.T
    df_coefs.columns = [model_name]
    # df_coefs.T.sort_values(by=0, ascending=False, inplace=True)
    return df_coefs
