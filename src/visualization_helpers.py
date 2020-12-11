#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Union

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import sklearn.metrics as mr
import sklearn.model_selection as ms
import sklearn.preprocessing as pp
import statsmodels.api as sm
from matplotlib import colors as mcolors
from matplotlib.ticker import FuncFormatter
from scipy.stats import norm
from sklearn.base import BaseEstimator
from yellowbrick.model_selection import LearningCurve
from yellowbrick.regressor import PredictionError, ResidualsPlot

SMALL_SIZE = 20
MEDIUM_SIZE = 22
BIGGER_SIZE = 24
plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rcParams["axes.facecolor"] = "white"
sns.set_style("darkgrid", {"legend.frameon": False})
sns.set_context("talk", font_scale=0.95, rc={"lines.linewidth": 2.5})


def show_yb_learning_curves(
    X,
    y,
    pipes,
    hspace=0.4,
    wspace=0.4,
    n_splits=5,
    n_repeats=5,
    scorer: Union[str, mr._scorer._PredictScorer] = "r2",
    save_plot: bool = False,
    fig_size: tuple = (12, 6),
    savefig: Path = Path().cwd() / "reports" / "figures",
):
    fig = plt.figure(constrained_layout=True, figsize=fig_size)
    gs = fig.add_gridspec(1, 2, hspace=hspace, wspace=wspace)
    ax1 = fig.add_subplot(gs[0, 0], sharey=None)
    ax2 = fig.add_subplot(gs[0, 1], sharey=None)
    for k, (pipe, ax) in enumerate(zip(pipes, [ax1, ax2])):
        visualizer = LearningCurve(
            pipe,
            scoring=scorer,
            cv=ms.RepeatedKFold(
                n_splits=n_splits, n_repeats=n_repeats, random_state=88
            ),
            ax=ax,
        )
        visualizer.fit(X, y)
        visualizer.finalize()
        if k == 1:
            ax.set_ylabel(None)
        ax.legend(loc="best", frameon=False)
    curr_datetime = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"pairplot__{curr_datetime}.png"
    if not (savefig / filename).is_file() and save_plot:
        fig.savefig(savefig / filename, bbox_inches="tight", dpi=300)


def do_nested_cv(
    est,
    X,
    y,
    num_trials,
    ocv,
    scorer="r2",
    outer_cv_n_splits=5,
    inner_cv_n_splits=5,
) -> List:
    nested_test_scores = np.zeros(num_trials)
    nested_train_scores = np.zeros(num_trials)
    nested_train_scores_min = np.zeros(num_trials)
    nested_train_scores_max = np.zeros(num_trials)
    nested_test_scores_min = np.zeros(num_trials)
    nested_test_scores_max = np.zeros(num_trials)
    for i in range(num_trials):
        inner_cv = ms.KFold(inner_cv_n_splits, shuffle=True, random_state=i)
        gs = ms.GridSearchCV(est, param_grid={}, cv=inner_cv, n_jobs=-1)
        if type(ocv).__name__ == "PredefinedSplit":
            outer_cv = ocv
        else:
            outer_cv = ms.KFold(
                n_splits=outer_cv_n_splits, shuffle=True, random_state=i
            )
        scores = ms.cross_validate(
            gs, X, y, scoring=scorer, cv=outer_cv, return_train_score=True
        )
        nested_train_scores[i] = scores["train_score"].mean()
        nested_test_scores[i] = scores["test_score"].mean()
        nested_train_scores_max[i] = scores["train_score"].max()
        nested_train_scores_min[i] = scores["train_score"].min()
        nested_test_scores_max[i] = scores["test_score"].max()
        nested_test_scores_min[i] = scores["test_score"].min()
    return [
        nested_train_scores,
        nested_test_scores,
        nested_test_scores_min,
        nested_test_scores_max,
        nested_train_scores_min,
        nested_train_scores_max,
    ]


def plot_nested_cv(
    est,
    X,
    y,
    num_nested_cv_trials: int,
    ps,
    scorer: str = "r2",
    outer_cv_n_splits: int = 5,
    inner_cv_n_splits: int = 5,
    hspace: float = 0.4,
    wspace: float = 0.4,
    fig_size: Tuple = (12, 8),
) -> None:
    fig = plt.figure(constrained_layout=True, figsize=fig_size)
    gs = fig.add_gridspec(2, 1, hspace=hspace, wspace=wspace)
    ax1 = fig.add_subplot(gs[0, 0], xticklabels=[])
    ax2 = fig.add_subplot(gs[1, 0])

    for outer_cv, ax in zip([ps, "kf"], [ax1, ax2]):
        train, test, test_min, test_max, train_min, train_max = do_nested_cv(
            est,
            X,
            y,
            num_nested_cv_trials,
            outer_cv,
            scorer,
            outer_cv_n_splits,
            inner_cv_n_splits,
        )
        pd.Series(train).plot(ax=ax, label="train", color="darkblue", lw=2.5)
        pd.Series(test).plot(ax=ax, label="test", color="red", lw=2.5)
        if outer_cv != "kf":
            ax.set_title(
                "Scores per Nested CV trial, with fixed test set",
                loc="left",
                fontweight="bold",
            )
        else:
            ax.set_title(
                "Scores per Nested CV trial, with variable test set",
                loc="left",
                fontweight="bold",
            )
        ax.set_ylabel(None)
        if outer_cv == "kf":
            ax.fill_between(
                range(num_nested_cv_trials),
                train_max,
                train_min,
                color="blue",
                label="train_range",
                lw=0,
                alpha=0.2,
            )
            ax.fill_between(
                range(num_nested_cv_trials),
                test_max,
                test_min,
                color="red",
                label="test_range",
                lw=0,
                alpha=0.2,
            )
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1), frameon=False)


def plot_multiple_histograms(
    X: pd.DataFrame,
    cols_to_plot: list,
    show_kde: bool = True,
    X_trans: pd.DataFrame = pd.DataFrame(),
    hspace: float = 0.2,
    wspace: float = 0.2,
    alpha: float = 0.95,
    save_plot: bool = False,
    fig_size: Tuple = (12, 6),
    savefig: Path = Path().cwd() / "reports" / "figures",
) -> None:
    fig = plt.figure(constrained_layout=True, figsize=fig_size)
    gs = fig.add_gridspec(2, len(cols_to_plot), hspace=hspace, wspace=wspace)
    for k, col in enumerate(cols_to_plot):
        ax1 = fig.add_subplot(gs[0, k], yticklabels=[])
        sns.histplot(X[col], kde=show_kde, ax=ax1)
        for tick in ax1.xaxis.get_major_ticks():
            tick.set_pad(-5)
        for tick in ax1.yaxis.get_major_ticks():
            tick.set_pad(-5)
        ax1.set_title(ax1.get_xlabel(), loc="left", fontweight="bold")
        ax1.set_xlabel(None)
        ci = norm(*norm.fit(X[col].dropna().to_numpy())).interval(alpha)
        plt.fill_betweenx(
            [0, ax1.get_ylim()[1]],
            ci[0],
            ci[1],
            color="g",
            label="95% c.i.",
            alpha=0.3,
        )
        ax1.legend(loc="upper left", bbox_to_anchor=(0.7, 1.15), frameon=False)

        if not X_trans.empty:
            ax2 = fig.add_subplot(gs[1, k], yticklabels=[])
            sns.histplot(X_trans[col], ax=ax2, kde=True)
            for tick in ax2.xaxis.get_major_ticks():
                tick.set_pad(-5)
            for tick in ax2.yaxis.get_major_ticks():
                tick.set_pad(-5)
            ax2.set_xlabel(None)
            ci = norm(*norm.fit(X_trans[col].dropna().to_numpy())).interval(
                alpha
            )
            plt.fill_betweenx(
                [0, ax2.get_ylim()[1]],
                ci[0],
                ci[1],
                color="g",
                label=f"{int(alpha*100)}% c.i.",
                alpha=0.3,
            )
    curr_datetime = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"multi_histograms__{curr_datetime}.png"
    if not (savefig / filename).is_file() and save_plot:
        fig.savefig(savefig / filename, bbox_inches="tight", dpi=300)


def plot_coef_plot(
    xvar: str,
    df: pd.DataFrame,
    fig_size: Tuple = (15, 12),
    save_fig: bool = False,
    savefig: Path = Path().cwd() / "reports" / "figures",
) -> None:
    ax = sns.barplot(x=xvar, y="feature", data=df, order=df["feature"])
    ax.set_xlabel("Coefficient importance")
    ax.set_title(ax.get_xlabel(), loc="left", fontweight="bold")
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.axvline(0, color="black", ls="-", lw=1.5)
    fig = plt.gcf()
    fig.set_figwidth(fig_size[0])
    fig.set_figheight(fig_size[1])
    for tick in ax.yaxis.get_major_ticks():
        tick.set_pad(-5)
    for tick in ax.xaxis.get_major_ticks():
        tick.set_pad(-5)
    curr_datetime = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"coefs__{curr_datetime}.png"
    if not (savefig / filename).is_file() and save_fig:
        fig.savefig(savefig / filename, bbox_inches="tight", dpi=300)


def plot_pairplot(
    df: pd.DataFrame,
    yvar: str,
    color_by_col: str,
    vars_to_plot: list,
    plot_specs: dict,
    wspace: float = 0.05,
    hspace: float = 0.05,
    save_plot: bool = False,
    fig_size: Tuple = (25, 25),
    savefig: Path = Path().cwd() / "reports" / "figures",
) -> None:
    """Generate a seaborn pairplot"""
    g = sns.pairplot(
        df,
        hue=color_by_col,
        x_vars=vars_to_plot,
        y_vars=yvar,
        plot_kws=plot_specs,
    )
    g.fig.set_figwidth(fig_size[0])
    g.fig.set_figheight(fig_size[1])
    fig = plt.gcf()
    fig.subplots_adjust(wspace=wspace, hspace=hspace)
    curr_datetime = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"pairplot__{curr_datetime}.png"
    if not (savefig / filename).is_file() and save_plot:
        fig.savefig(savefig / filename, bbox_inches="tight", dpi=300)


def plot_corr_map(
    df: pd.DataFrame,
    annot: bool = False,
    annot_fmt: str = ".2g",
    square: bool = False,
    color_bar: bool = False,
    x_axis_ticks_angle: int = 45,
    x_axis_ticks_horizontal_alignment: str = "right",
    color_bar_kws: Dict = {"shrink": 0.99},
    fig_size: Tuple = (15, 12),
    save_fig: bool = False,
    savefig: Path = Path().cwd() / "reports" / "figures",
) -> None:
    df_corr = df.corr()
    fig, ax = plt.subplots(figsize=fig_size)
    ax = sns.heatmap(
        df[df_corr.sum().sort_values(ascending=False).index.values].corr(),
        mask=np.triu(df_corr),
        cmap=sns.diverging_palette(220, 10, as_cmap=True),
        vmax=1,
        vmin=-1,
        center=0,
        square=square,
        linewidths=0.5,
        annot=annot,
        fmt=annot_fmt,
        cbar=color_bar,
        cbar_kws=color_bar_kws,
    )
    ax.set_facecolor("white")
    _ = ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=x_axis_ticks_angle,
        ha=x_axis_ticks_horizontal_alignment,
        fontsize=14,
    )
    _ = ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=14)
    curr_datetime = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"corr_heatmap__{curr_datetime}.png"
    if not (savefig / filename).is_file() and save_fig:
        fig.savefig(savefig / filename, bbox_inches="tight", dpi=300)


def plot_qq(
    res: pd.Series,
    fig_size: Tuple = (15, 12),
    save_fig: bool = False,
    savefig: Path = Path().cwd() / "reports" / "figures",
) -> None:
    """Plot Q-Q (quantile-quantile) plot"""
    fig, ax = plt.subplots(figsize=fig_size)
    sm.ProbPlot(data=res, dist=stats.distributions.norm, fit=True).qqplot(
        line="s", ax=ax
    )
    ax.get_lines()[0].set_markersize(10)
    ax.set_title("Normal Q-Q plot", fontsize=16)
    curr_datetime = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"qq_plot__{curr_datetime}.png"
    if not (savefig / filename).is_file() and save_fig:
        fig.savefig(savefig / filename, bbox_inches="tight", dpi=300)


def plot_cv_scores(
    cv_results: dict,
    cols: List,
    fig_size: Tuple = (8, 8),
    save_fig: bool = False,
    savefig: Path = Path().cwd() / "reports" / "figures",
) -> None:
    """Plot linear model coefficients."""
    coefs = pd.DataFrame(
        [est.named_steps["reg"].coef_ for est in cv_results["estimator"]],
        columns=cols,
    )
    fig, _ = plt.subplots(figsize=fig_size)
    ax = sns.swarmplot(data=coefs, orient="h", color="k", alpha=0.5)
    ax = sns.boxplot(data=coefs, orient="h", color="cyan", saturation=0.5)
    ax.axvline(x=0, color="0.5")
    ax.set_xlabel("Coefficient importance")
    ax.set_title("Coefficient importance and its variability")
    curr_datetime = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"qq_plot__{curr_datetime}.png"
    if not (savefig / filename).is_file() and save_fig:
        fig.savefig(savefig / filename, bbox_inches="tight", dpi=300)


def plot_residual_manually(
    estimator: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    fig_size: Tuple = (15, 12),
    save_fig: bool = False,
    savefig: Path = Path().cwd() / "reports" / "figures",
) -> list:
    """Graph fit residual"""
    estimator.fit(X_train, y_train)
    y_pred = pd.Series(estimator.predict(X_test), index=y_test.index)
    fig, ax = plt.subplots(figsize=(15, 12))
    res = y_pred - y_test
    ax.scatter(x=y_pred, y=res, c="black", s=80, edgecolor="white")
    ax.set_title(f"Residuals_{type(estimator.named_steps['reg']).__name__}")
    ax.set_xlabel("Predicted Value")
    ax.set_ylabel("Residuals")
    curr_datetime = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"Residual_plot__{curr_datetime}.png"
    if not (savefig / filename).is_file() and save_fig:
        fig.savefig(savefig / filename, bbox_inches="tight", dpi=300)

    y_comparison = pd.DataFrame()
    y_comparison["pred"] = pd.Series(y_pred, index=y_test.index)
    y_comparison["test"] = y_test
    y_comparison["res"] = res
    X_comp = X_test.merge(y_comparison, left_index=True, right_index=True)
    return [X_comp, estimator, res]


def show_yb_prediction_error(
    estimator,
    X,
    y,
    axis_label_fontsize=20,
    axis_tick_label_pad=5,
    fig_size: Tuple = (15, 12),
    save_fig: bool = False,
    savefig: Path = Path().cwd() / "reports" / "figures",
) -> None:
    """Show plot of regression prediction error"""
    fig, ax = plt.subplots(figsize=(15.5, 12))
    visualizer = PredictionError(
        estimator,
        is_fitted="auto",
        ax=ax,
        legend=False,
        alpha=1.0,
        shared_limits=False,
    )
    visualizer.fit(X, y)
    visualizer.score(X, y)
    visualizer.finalize()
    ax.get_xaxis().set_major_formatter(
        FuncFormatter(lambda x, p: format(int(x), ","))
    )
    ax.get_yaxis().set_major_formatter(
        FuncFormatter(lambda x, p: format(int(x), ","))
    )
    ax.legend(frameon=False)
    ax.set_xlabel(
        "Observed Price ($)",
        fontsize=axis_label_fontsize,
    )
    ax.set_ylabel(
        "Predicted Price ($)",
        fontsize=axis_label_fontsize,
    )
    ax.tick_params(which="both", direction="out", pad=axis_tick_label_pad)
    ax.set_title(
        f"Prediction Error for {type(estimator.named_steps['reg']).__name__}",
        loc="center",
        fontweight="bold",
    )
    curr_datetime = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"yb_pred_error__{curr_datetime}.png"
    if not (savefig / filename).is_file() and save_fig:
        fig.savefig(savefig / filename, bbox_inches="tight", dpi=300)


def show_yb_residual_plot(
    estimator,
    X_train,
    y_train,
    X_test,
    y_test,
    fig_size: Tuple = (15, 12),
    save_fig: bool = False,
    savefig: Path = Path().cwd() / "reports" / "figures",
) -> None:
    """Show plot of test residuals."""
    y_pred = estimator.predict(X_test)
    mse = mr.mean_squared_error(y_test, y_pred, squared=False)
    mae = mr.mean_absolute_error(y_test, y_pred)
    mdae = mr.median_absolute_error(y_test, y_pred)
    lines = {
        "MAE": [mae, "red"],
        "MDAE": [mdae, "darkred"],
        "RMSE": [mse, "magenta"],
    }

    fig, ax = plt.subplots(figsize=(15.5, 12))
    visualizer = ResidualsPlot(
        estimator, is_fitted="auto", ax=ax, legend=False, test_color="teal"
    )
    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)
    visualizer.finalize()

    for k, v in lines.items():
        ax.axhline(y=0 + v[0], alpha=0.5, linestyle="--", color=v[1], label=k)
        ax.axhline(
            y=0 - v[0], alpha=0.5, linestyle="--", color=v[1], label=None
        )

    # handles, labels = ax.get_legend_handles_labels()
    # h_s, k_s = [[] for _ in range(2)]
    # for key, handle in zip(labels, handles):
    #     if any(ext in key for ext in list(lines.keys())):
    #         h_s.append(handle)
    #         k_s.append(key)
    # ax.legend(h_s, k_s)

    curr_datetime = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"yb_resid_plot__{curr_datetime}.png"
    if not (savefig / filename).is_file() and save_fig:
        fig.savefig(savefig / filename, bbox_inches="tight", dpi=300)


def plot_multi_feature_target(
    df,
    tooltip_list,
    rows,
    columns,
    title_font_size,
    label_font_size,
    marker_size,
    marker_opacity,
    marker_linewidth,
    color_by_col,
    legend_offset=-5,
    save_fig: bool = False,
    savefig: Path = Path().cwd() / "reports" / "figures",
    fig_size: Tuple = (250, 300),
) -> alt.Chart:
    chart = (
        alt.Chart(df)
        .mark_circle(
            size=marker_size,
            strokeWidth=marker_linewidth,
            stroke="white",
            strokeOpacity=1,
            opacity=marker_opacity,
            clip=True,
        )
        .encode(
            alt.X(
                alt.repeat("row"),
                # title="",
                type="quantitative",
                axis=alt.Axis(
                    labels=True,
                    domainWidth=2,
                    domainColor="black",
                    domainOpacity=1,
                ),
            ),
            alt.Y(
                alt.repeat("column"),
                # title="",
                type="quantitative",
                axis=alt.Axis(
                    labels=True,
                    domainWidth=2,
                    domainColor="black",
                    domainOpacity=1,
                ),
            ),
            color=alt.Color(
                color_by_col,
                legend=alt.Legend(
                    # columns=df[color_by_col].nunique(),
                    # orient="top",
                    # titleOrient="left",
                    offset=legend_offset,
                    titleFontSize=title_font_size,
                ),
            ),
            tooltip=tooltip_list,
        )
        .properties(width=fig_size[0], height=fig_size[1])
        .repeat(row=rows, column=columns, spacing=0)
        .configure_axis(
            labelFontSize=label_font_size, titleFontSize=title_font_size
        )
        .interactive()
    )
    curr_datetime = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"multi_feature_plot__{curr_datetime}.html"
    if not (savefig / filename).is_file() and save_fig:
        chart.save(str(savefig / filename), format="html")
    return chart


def plot_single_feature_target(
    df,
    x,
    y,
    tooltip_list,
    cats_cols_bar_chart,
    title_font_size,
    label_font_size,
    marker_size,
    marker_opacity,
    marker_linewidth,
    color_scheme,
    title_horizontal_offset=45,
    title_vertical_offset=-3,
    save_fig: bool = False,
    savefig: Path = Path().cwd() / "reports" / "figures",
    fig_size: Tuple = (300, 250),
) -> alt.Chart:
    figs = {}
    for ax_row, col in enumerate(cats_cols_bar_chart):
        ptitle = f"{y} vs {x}" if ax_row == 0 else ""
        ncols = 2 if df[col].nunique() > 10 else 1
        col_chart = (
            alt.Chart(df, title=ptitle)
            .mark_circle(
                size=marker_size,
                strokeWidth=marker_linewidth,
                stroke="white",
                strokeOpacity=1,
                opacity=marker_opacity,
            )
            .encode(
                alt.X(
                    f"{x}:Q",
                    axis=alt.Axis(
                        title="",
                        labels=True,
                        domainWidth=2,
                        domainColor="black",
                        domainOpacity=1,
                    ),
                ),
                alt.Y(
                    f"{y}:Q",
                    axis=alt.Axis(
                        title="",
                        labels=True,
                        domainWidth=2,
                        domainColor="black",
                        domainOpacity=1,
                    ),
                ),
                color=alt.Color(
                    f"{col}:N",
                    legend=alt.Legend(columns=ncols),
                    scale=alt.Scale(scheme=color_scheme),
                ),
                tooltip=tooltip_list + [col],
            )
            .properties(width=fig_size[0], height=fig_size[1])
        )
        figs[col] = col_chart
    combined_chart = (
        alt.vconcat(
            alt.hconcat(
                figs[cats_cols_bar_chart[0]], figs[cats_cols_bar_chart[1]]
            ).resolve_scale(color="independent"),
            alt.hconcat(
                figs[cats_cols_bar_chart[2]], figs[cats_cols_bar_chart[3]]
            ).resolve_scale(color="independent"),
            alt.hconcat(
                figs[cats_cols_bar_chart[4]], figs[cats_cols_bar_chart[5]]
            ).resolve_scale(color="independent"),
            alt.hconcat(
                figs[cats_cols_bar_chart[6]], figs[cats_cols_bar_chart[7]]
            ).resolve_scale(color="independent"),
            alt.hconcat(
                figs[cats_cols_bar_chart[8]], figs[cats_cols_bar_chart[9]]
            ).resolve_scale(color="independent"),
            alt.hconcat(
                figs[cats_cols_bar_chart[10]], figs[cats_cols_bar_chart[11]]
            ).resolve_scale(color="independent"),
            figs[cats_cols_bar_chart[12]],
        )
        .resolve_scale(color="independent")
        .configure_title(
            anchor="start",
            dx=title_horizontal_offset,
            offset=title_vertical_offset,
            fontSize=title_font_size + 2,
        )
        .configure_axis(
            labelFontSize=label_font_size, titleFontSize=title_font_size
        )
    )
    curr_datetime = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"single_feature_plot__{curr_datetime}.html"
    if not (savefig / filename).is_file() and save_fig:
        combined_chart.save(str(savefig / filename), format="html")
    return combined_chart


def plot_multiple_bar_charts(
    df,
    cats_cols_bar_chart,
    annotation_horiz=20,
    annotation_vert=3,
    title_vertical_offset=-3,
    label_font_size=12,
    title_font_size=12,
    fig_size: Tuple = (250, 350),
):
    figs = {}
    for col in cats_cols_bar_chart:
        col_chart = (
            alt.Chart(df, title=col)
            .mark_bar()
            .encode(
                alt.Y(
                    f"{col}:N",
                    axis=alt.Axis(
                        title="",
                        labels=True,
                        domainWidth=2,
                        domainColor="black",
                        domainOpacity=1,
                    ),
                    sort="-x",
                ),
                alt.X(
                    f"count({col}):Q",
                    axis=alt.Axis(
                        title="",
                        labels=True,
                        domainWidth=2,
                        domainColor="black",
                        domainOpacity=1,
                    ),
                ),
                tooltip=[alt.Tooltip(f"count({col}):Q", title="Number")],
            )
            .properties(width=fig_size[0], height=fig_size[1])
        )
        text = (
            alt.Chart(df)
            .mark_text(dx=annotation_horiz, dy=annotation_vert, color="black")
            .encode(
                x=alt.X(f"count({col}):Q"),
                y=alt.Y(f"{col}:N", sort="-x"),
                text=alt.Text(f"count({col}):Q", format=".0d"),
            )
        )
        figs[col] = col_chart + text
    combined_chart = (
        alt.hconcat(
            figs[cats_cols_bar_chart[0]],
            figs[cats_cols_bar_chart[1]],
            figs[cats_cols_bar_chart[2]],
            figs[cats_cols_bar_chart[3]],
            figs[cats_cols_bar_chart[4]],
            figs[cats_cols_bar_chart[5]],
            figs[cats_cols_bar_chart[6]],
            figs[cats_cols_bar_chart[7]],
            figs[cats_cols_bar_chart[0]],
            figs[cats_cols_bar_chart[1]],
            figs[cats_cols_bar_chart[2]],
            figs[cats_cols_bar_chart[11]],
        )
        .resolve_scale(color="independent")
        .configure_title(
            anchor="middle",
            offset=title_vertical_offset,
            fontSize=title_font_size + 2,
        )
        .configure_axis(
            labelFontSize=label_font_size, titleFontSize=title_font_size
        )
    )
    return combined_chart


def plot_side_by_side_bar_box(
    df_coeffs,
    xvar,
    df_perm_imp,
    box_sort,
    neg_bar_color="darkred",
    pos_bar_color="blue",
    box_size=25,
    box_color="blue",
    split_type="train",
    title_font_size=12,
    label_font_size=12,
    title_horizontal_offset=115,
    title_vertical_offset=-3,
    show_bar_chart=True,
    save_fig: bool = False,
    savefig: Path = Path().cwd() / "reports" / "figures",
    fig_size: Tuple = (350, 200),
):
    lin_coeffs = (
        alt.Chart(df_coeffs, title="Model Coefficients")
        .mark_bar()
        .encode(
            x=alt.X(
                f"{xvar}:Q",
                axis=alt.Axis(
                    title="",
                    labels=True,
                    domainWidth=2,
                    domainColor="black",
                    domainOpacity=1,
                ),
            ),
            y=alt.Y(
                "feature:N",
                sort=None,
                axis=alt.Axis(
                    title="",
                    labels=True,
                    domainWidth=2,
                    domainColor="black",
                    domainOpacity=1,
                ),
            ),
            color=alt.condition(
                alt.datum[xvar] > 0,
                alt.value(pos_bar_color),
                alt.value(neg_bar_color),
            ),
        )
        .properties(width=fig_size[0], height=fig_size[1])
    )
    perm_imp = (
        alt.Chart(
            df_perm_imp, title=f"Permutation importances ({split_type} set)"
        )
        .mark_boxplot(color=box_color, size=box_size)
        .encode(
            x=alt.X(
                "importance:Q",
                axis=alt.Axis(
                    title="",
                    labels=True,
                    domainWidth=2,
                    domainColor="black",
                    domainOpacity=1,
                ),
            ),
            y=alt.Y(
                "feature:N",
                axis=alt.Axis(
                    title="",
                    labels=True,
                    domainWidth=2,
                    domainColor="black",
                    domainOpacity=1,
                ),
                title="",
                sort=box_sort,
            ),
        )
        .properties(width=fig_size[0], height=fig_size[1])
    )
    if show_bar_chart:
        combined_chart = alt.hconcat(lin_coeffs, perm_imp)
    else:
        combined_chart = perm_imp
    combined_chart = combined_chart.configure_title(
        anchor="start",
        dx=title_horizontal_offset,
        offset=title_vertical_offset,
        fontSize=title_font_size,
    ).configure_axis(
        labelFontSize=label_font_size, titleFontSize=title_font_size
    )
    curr_datetime = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"bar_box_plot_{split_type}__{curr_datetime}.html"
    if not (savefig / filename).is_file() and save_fig:
        combined_chart.save(str(savefig / filename), format="html")
    return combined_chart


def show_3d_plot(df, xcol, ycol, zcol, ptitle, fig_size=(8, 8)):
    cmap = mcolors.LinearSegmentedColormap.from_list(
        name="green_white_red",
        colors=[(1, 0, 0), (1, 1.0, 1), (0, 1, 0)],
        N=len(df) - 1,
    )
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection="3d")
    _ = ax.scatter(
        df[xcol],
        df[ycol],
        df[zcol],
        c=df[0],
        cmap=cmap,
        marker="o",
        vmin=df[0].min(),
        vmax=df[0].max(),
    )
    ax.grid()
    ax.set_xlabel("Under-Prediction Penalty")
    ax.set_ylabel("Over-Prediction Penalty")
    ax.set_zlabel("Within threshold Reward")
    ax.set_title(ptitle, loc="left", fontweight="bold")


def plot_single_feature_multi_transformations(
    series,
    colname,
    hspace=0.075,
    save_fig: bool = False,
    savefig: Path = Path().cwd() / "reports" / "figures",
    fig_size=(12, 16),
):
    fig = plt.figure(figsize=fig_size)
    grid = plt.GridSpec(3, 1, hspace=hspace)
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[1, 0])
    ax3 = fig.add_subplot(grid[2, 0])
    sns.histplot(
        series,
        ax=ax1,
        kde=True,
        color="steelblue",
        label="Raw",
        alpha=0.75,
    )
    sns.histplot(
        np.log(series),
        ax=ax2,
        kde=True,
        label="Log",
        color="darkred",
        alpha=0.45,
    )
    sns.histplot(
        pp.PowerTransformer(method="yeo-johnson")
        .fit_transform(series.to_frame())
        .squeeze(),
        ax=ax3,
        alpha=0.4,
        color="red",
        kde=True,
        label="Yeo-Johnson",
    )
    sns.histplot(
        pp.PowerTransformer(method="box-cox")
        .fit_transform(series.to_frame())
        .squeeze(),
        ax=ax3,
        alpha=0.4,
        color="yellow",
        ls="-.",
        kde=True,
        label="Box-Cox",
    )
    sns.histplot(
        pp.QuantileTransformer(output_distribution="normal")
        .fit_transform(series.to_frame())
        .squeeze(),
        ax=ax3,
        alpha=0.35,
        kde=True,
        color="blue",
        label="Quantile",
    )
    ax1.set_title(
        f"(Count) Histogram of {colname}", loc="left", fontweight="bold"
    )
    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel(None)
        ax.legend()
        ax.set_ylabel(None)
        ax.tick_params(pad=-5, axis="both", labelsize=20)
    curr_datetime = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"single_feature_multi_transformations__{curr_datetime}.png"
    if not (savefig / filename).is_file() and save_fig:
        fig.savefig(savefig / filename, bbox_inches="tight", dpi=300)


def plot_scatter_plot_grouped(
    df,
    x,
    y,
    color_by_col,
    x_title,
    y_title,
    ptitle_cname,
    save_fig: bool = False,
    savefig: Path = Path().cwd() / "reports" / "figures",
    fig_size=(10, 10),
):
    fig, ax = plt.subplots(figsize=fig_size)
    sns.scatterplot(
        x=x,
        y=y,
        hue=color_by_col,
        palette=["purple", "darkorange"],
        data=df,
        ax=ax,
        s=80,
    )
    ax.set_xlabel(x_title)
    ax.set_ylabel(None)
    ax.legend(
        loc="upper left",
        ncol=2,
        bbox_to_anchor=(0.75, 1.06),
        handletextpad=0.025,
        columnspacing=0.1,
    )
    ax.get_xaxis().set_major_formatter(
        FuncFormatter(lambda x, p: format(int(x), ","))
    )
    ax.get_yaxis().set_major_formatter(
        FuncFormatter(lambda x, p: format(int(x), ","))
    )
    ptitle = f"{y_title}, by {ptitle_cname}"
    ax.set_title(ptitle, loc="left", fontweight="bold")
    curr_datetime = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{y_title}_by_{x_title}__{curr_datetime}.png"
    if not (savefig / filename).is_file() and save_fig:
        fig.savefig(savefig / filename, bbox_inches="tight", dpi=300)


def plot_multiple_histograms_v2(
    ss,
    snames,
    colors,
    colname,
    save_fig: bool = False,
    savefig: Path = Path().cwd() / "reports" / "figures",
    fig_size=(8, 6),
):
    fig, ax = plt.subplots(figsize=fig_size)
    for s, sname, color in zip(ss, snames, colors):
        sns.histplot(
            s,
            ax=ax,
            alpha=0.45,
            kde=True,
            color=color,
            label=sname,
        )
    ax.set_title(
        f"(Count) Histogram of {colname}", loc="left", fontweight="bold"
    )
    ax.legend()
    ax.get_xaxis().set_major_formatter(
        FuncFormatter(lambda x, p: format(int(x), ","))
    )
    ax.set_xlabel("Price ($)")
    curr_datetime = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"target_histogram__{curr_datetime}.png"
    if not (savefig / filename).is_file() and save_fig:
        fig.savefig(savefig / filename, bbox_inches="tight", dpi=300)
