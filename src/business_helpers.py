#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd


def calculate_cost(
    relative_pens,
    threshold,
    fit_residual,
    fit_residual_dummy_median,
    fit_residual_dummy_mean,
    y_test,
):
    d_costs_to_client = {}
    for preds_res, m_type in zip(
        [fit_residual, fit_residual_dummy_median, fit_residual_dummy_mean],
        ["ML", "Naive_Median", "Naive_Mean"],
    ):
        under_estimates = preds_res.loc[
            (preds_res < 0) & (preds_res.abs() > threshold)
        ]
        over_estimates = preds_res.loc[
            (preds_res > 0) & (preds_res.abs() > threshold)
        ]
        within_threshold = pd.Series(y_test, index=fit_residual.index).loc[
            preds_res.abs() <= threshold
        ]
        under_est_pen_frac = (
            under_estimates.abs() * relative_pens["under-estimate-penalty"]
        )
        under_est_pen = (under_est_pen_frac / 100).reset_index(drop=True)
        over_est_pen_frac = (
            over_estimates.abs() * relative_pens["over-estimate-penalty"]
        )
        over_est_pen = (over_est_pen_frac / 100).reset_index(drop=True)
        thresh_pen_frac = within_threshold * relative_pens["equality"]
        thresh_pen = (thresh_pen_frac / 100).reset_index(drop=True)
        preds_res_penalized = under_est_pen + over_est_pen + thresh_pen
        d_costs_to_client[m_type] = {
            "median": preds_res_penalized.median(),
            "sum": preds_res_penalized.sum(),
        }
    df_costs_to_client = pd.DataFrame.from_dict(
        d_costs_to_client, orient="index"
    )
    return df_costs_to_client


def calculate_grid_of_penalties(
    penalties_grid,
    threshold,
    fit_residual,
    fit_residual_dummy_median,
    fit_residual_dummy_mean,
    y_test,
):
    dfs_cost_to_client = {}
    for u in list(penalties_grid["u"]):
        for o in list(penalties_grid["o"]):
            for e in list(penalties_grid["e"]):
                df_cost_to_client = calculate_cost(
                    {
                        "over-estimate-penalty": -o,
                        "under-estimate-penalty": -u,
                        "equality": e,
                    },
                    threshold,
                    fit_residual,
                    fit_residual_dummy_median,
                    fit_residual_dummy_mean,
                    y_test,
                )
                dfs_cost_to_client[f"{u}_{o}_{e}"] = df_cost_to_client.loc[
                    "ML", "median"
                ]
    df_costs = pd.DataFrame.from_dict(
        dfs_cost_to_client, orient="index"
    ).reset_index()
    df_costs = (
        df_costs.merge(
            df_costs["index"]
            .str.split("_", expand=True)
            .rename(columns={0: "u", 1: "o", 2: "e"}),
            left_index=True,
            right_index=True,
        )
        .set_index("index")
        .astype(float)
    )
    return df_costs
