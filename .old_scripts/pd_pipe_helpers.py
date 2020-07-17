#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd

from src.ml_helperrs import filter_by_value_counts


def drop_dups(df, keep="first"):
    return df.drop_duplicates(keep=keep)


def drop_zero_per_month_min(df):
    return df[df["per_month_min"] != 0]


def col_filter(df, col, col_threshold: int = 80):
    return df[df[col] < col_threshold]


def mpg_polynomial_transformer(df, col, polynomial_degree: float = 0.5):
    deg = (
        f"{polynomial_degree.replace('.', '')}"
        if type(polynomial_degree) == float
        else polynomial_degree
    )
    df[deg] = df[col] ** polynomial_degree
    return df


def interaction_terms(df, f1, f2):
    df[f"{f1}_x_{f2}"] = df[f1] * df[f2]
    return df


def change_col_dtype(df, col_name, new_dtype: str = "str"):
    df[col_name] = df[col_name].astype(new_dtype)


def df_feature_generator(df: pd.DataFrame) -> pd.DataFrame:
    df = (
        df.pipe(drop_dups, keep="first")
        .pipe(drop_zero_per_month_min)
        .pipe(col_filter, col="MPG", col_threshold=80)
        .pipe(mpg_polynomial_transformer, col="MPG", polynomial_degree=0.5)
        .pipe(interaction_terms, f1="MPG05", f2="tankvolume")
        .pipe(interaction_terms, f1="Mileage", f2="tankvolume")
        .pipe(interaction_terms, f1="MPG05", f2="consumer_reviews")
        .pipe(col_filter, col="Mileage", col_threshold=10000)
        .pipe(
            filter_by_value_counts,
            colname="Fuel Type",
            mask=df["Fuel Type"].value_counts()[
                df["Fuel Type"].value_counts() < 10
            ],
        )
        .pipe(
            filter_by_value_counts,
            colname="trans_speed",
            mask=df["trans_speed"].value_counts()[
                df["trans_speed"].value_counts() < 10
            ],
        )
        .pipe(
            filter_by_value_counts,
            colname="make",
            mask=df["make"].value_counts()[df["make"].value_counts() < 10],
        )
        .pipe(change_col_dtype, col_name="consumer_stars", new_dtype="str")
        .pipe(change_col_dtype, col_name="seller_rating", new_dtype="str")
        .pipe(
            filter_by_value_counts,
            colname="seller_rating",
            mask=df["seller_rating"].value_counts()[
                (
                    df["seller_rating"]
                    .value_counts()
                    .index.isin([3.8, 3.7, 4.0])
                ).tolist()
            ],
        )
        .pipe(
            filter_by_value_counts,
            colname="consumer_stars",
            mask=df["consumer_stars"].value_counts()[
                df["consumer_stars"].value_counts() < 10
            ],
        )
    )
    return df
