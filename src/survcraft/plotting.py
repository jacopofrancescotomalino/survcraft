r"""Plot survival distributions and recorded training losses.

Notes
-----
Requires the plotting extras: pandas, matplotlib, and seaborn. Functions
create matplotlib figures as side effects and return None unless stated
otherwise. Model data and prediction contracts are in SurvivalEstimator.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from itertools import cycle
from matplotlib import patheffects
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from sksurv.nonparametric import kaplan_meier_estimator

from .util import detect_max_survival_time

def _extract_times_events(y):
    """Return (times, events) from structured or 2-column targets."""
    if y is None:
        return None, None

    if getattr(y, "dtype", None) is not None and y.dtype.names:
        time_name = "time" if "time" in y.dtype.names else y.dtype.names[1]
        event_name = "event" if "event" in y.dtype.names else y.dtype.names[0]
        return y[time_name], y[event_name]

    return y[:, 0], y[:, 1]


def _get_colors(n_samples):
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
    cycler = cycle(color_cycle)
    return [next(cycler) for _ in range(n_samples)]


def plot_outputs(model, X, y = None, max_time=None):
    # time = np.linspace(0.0, times_train[events_train].max(), 101)
    if max_time is None:
        max_time = detect_max_survival_time(model, X)
        print("estimated max time:", max_time)  # time[-1].item())
    time_grid = np.linspace(0.0, max_time, 101)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10), dpi = 100)
    n_samples = X.shape[0]
    # Pre-generate colors for consistent coloring across plots
    alpha_opt = 1/np.sqrt(n_samples)

    # plots histogram of observations (only if y is provided)
    if y is not None:
        colors = _get_colors(n_samples)
        times, events = _extract_times_events(y)
        data = list(zip(events, times))
        observed_times = [data[1] for data in data if data[0] == True]
        censored_times = [data[1] for data in data if data[0] == False]
        axes[0][0].hist(observed_times, bins=30, alpha=0.5, label='Observed', color='steelblue')
        axes[0][0].hist(censored_times, bins=30, alpha=0.5, label='Censored', color='salmon')
        axes[0][0].set_xlabel("Time")
        axes[0][0].set_ylabel("Count")
        axes[0][0].set_xlim(0, max_time)
        axes[0][0].set_title("histogram of observations")
        axes[0][0].legend()
    

    # Plot curves with optional vertical lines for observed event times (color each curve according to the observed time and event status with the viridis colormap)
    survival_preds = model.predict("survival", X, time_grid)
    density_preds = model.predict("density", X, time_grid)
    # Hazard can be legitimately infinite at t=0 (e.g. Weibull with shape < 1),
    # which predict rejects; evaluate from the second grid point and leave t=0
    # as NaN so matplotlib skips it.
    hazard_tail = model.predict("hazard", X, time_grid[1:])
    hazard_preds = np.concatenate(
        [np.full((hazard_tail.shape[0], 1), np.nan, dtype=hazard_tail.dtype), hazard_tail], axis=1
    )

    if y is not None:
        times, events = _extract_times_events(y)
        cmap = plt.get_cmap('viridis')
        max_obs_time = max(times)
        min_obs_time = min(times)
        colors = [cmap((t - min_obs_time) / (max_obs_time - min_obs_time)) for t in times]
        # show colorbar for the survival plot
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(min_obs_time, max_obs_time))
        sm.set_array([])
        #align the colorbar with the complete plot, not only one of the subplots
        cbar = plt.colorbar(sm, ax = axes[0][1], fraction=0.02, pad=0.04)
        cbar.set_label('Observed Time')
        alpha_values = [alpha_opt if e else alpha_opt / 2 for e in events]
    else :
        colors = _get_colors(n_samples)
        alpha_values = [alpha_opt for e in range(n_samples)]
    
    for hazard, dens, surv, color, a_value in zip(hazard_preds, density_preds, survival_preds, colors, alpha_values):
        axes[0][1].plot(time_grid, surv, color=color, alpha=a_value)
        axes[1][0].plot(time_grid, dens, color=color, alpha=a_value)
        axes[1][1].plot(time_grid, hazard, color=color, alpha=a_value)

    axes[0][1].set_title("survival")
    axes[1][0].set_title("density")
    axes[1][1].set_title("hazard")
    axes[0][1].set_xlabel("Time")
    axes[0][1].set_ylabel("Survival Probability")
    axes[1][0].set_xlabel("Time")
    axes[1][0].set_ylabel("Density")
    axes[1][1].set_xlabel("Time")
    axes[1][1].set_ylabel("Hazard")
    #axes[1][0].set_yscale("log")
    #axes[1][1].set_yscale("log")
    axes[0][1].set_xlim(0, max_time)
    axes[1][0].set_xlim(0, max_time)
    axes[1][1].set_xlim(0, max_time)

    event_shape_dict = {True : 'X', False : 'P'}

    if y is not None :
        for index, obs_time, obs_event, color, surv, dens, hazard in zip(range(n_samples), times, events, colors, survival_preds, density_preds, hazard_preds):
            clipped_time = min(time_grid[-1], obs_time)
            # Interpolate curve value at observed time to place one marker per curve.
            surv_at_t = np.interp(clipped_time, time_grid, surv)
            dens_at_t = np.interp(clipped_time, time_grid, dens)
            hazard_at_t = np.interp(clipped_time, time_grid, hazard)
            #axes[0][1].axvline(clipped_time, linestyle="solid" if obs_event else "dotted", color=color, alpha=alpha_opt)
            if obs_event == False:
                axes[0][1].scatter(clipped_time, surv_at_t, color=color, marker=event_shape_dict[obs_event], alpha=0.2)
            #axes[1][0].axvline(clipped_time, linestyle="solid" if obs_event else "dotted", color=color, alpha=alpha_opt)
            if obs_event == True:
                axes[1][0].scatter(clipped_time, dens_at_t, color=color, marker=event_shape_dict[obs_event], alpha=0.2)
                #axes[1][1].axvline(clipped_time, linestyle="solid" if obs_event else "dotted", color=color, alpha=alpha_opt)
                axes[1][1].scatter(clipped_time, hazard_at_t, color=color, marker=event_shape_dict[obs_event], alpha=0.2)
    return

def _draw_group_curves(ax, survival, time_grid, color, n_std, show_ci, individual_alpha,
                       times, events, km_color, km_ci):
    """Draw the expected survival of one group and, if times is given, its Kaplan-Meier estimate."""
    

    # Survivors at t are a sum of independent Bernoulli(S_i(t)).
    mean = survival.mean(axis=0)
    std = np.sqrt(np.sum(survival * (1 - survival), axis=0)) / len(survival)
    if show_ci:
        ax.fill_between(time_grid, np.clip(mean - n_std * std, 0, 1), np.clip(mean + n_std * std, 0, 1),
                        color=color, alpha=0.2, linewidth=0, zorder=1)
    mean_effects = None
    if individual_alpha is not None:
        ax.add_collection(LineCollection([np.column_stack([time_grid, s]) for s in survival],
                                         colors=color, linewidths=0.6, alpha=individual_alpha, zorder=1))
        # Ring in the background color keeps the mean visible over the individual curves.
        mean_effects = [patheffects.Stroke(linewidth=5, foreground=ax.get_facecolor()), patheffects.Normal()]
    ax.plot(time_grid, mean, color=color, linewidth=2.5, zorder=5, path_effects=mean_effects)

    if times is not None:
        km_times, km_surv, km_conf = kaplan_meier_estimator(events, times, conf_type="log-log")
        # Start the step function at S(0) = 1.
        km_times, km_surv = np.r_[0.0, km_times], np.r_[1.0, km_surv]
        ax.step(km_times, km_surv, where="post", color=km_color, linewidth=2, linestyle="--", zorder=6)
        if km_ci:
            ax.fill_between(km_times, np.r_[1.0, km_conf[0]], np.r_[1.0, km_conf[1]],
                            step="post", color=km_color, alpha=0.15, linewidth=0, zorder=2)


def plot_expected_survival(model, X, partition=None, y=None, time_grid=None,
                           show_individual=False, show_ci=True, show_km=True, km_ci=True,
                           n_std=1.96, n_cols=4):
    r"""Plot the expected survival of groups of samples against their Kaplan-Meier estimates.

    The expected survival of a group is the mean of the predicted survival
    curves of its samples, i.e. the Kaplan-Meier curve the model expects for
    that group.

    Parameters
    ----------
    model : SurvivalEstimator
        Initialized estimator supporting survival prediction.
    X : numpy.ndarray, shape (n_samples, n_features)
        Features of the samples.
    partition : numpy.ndarray of int, shape (n_samples,), optional
        Group label of each sample. Defaults to a single group.
    y : structured array or numpy.ndarray of shape (n_samples, 2), optional
        Observed outcomes of the samples, used for the Kaplan-Meier estimates.
    time_grid : numpy.ndarray, optional
        Times where predictions are evaluated. Defaults to 200 points from 0 to
        the largest observed time, or to detect_max_survival_time without y.
    show_individual : bool, default=False
        Also draw the predicted survival curve of each sample.
    show_ci : bool, default=True
        Draw a band of n_std standard deviations around the expected survival.
    show_km : bool, default=True
        Draw the Kaplan-Meier estimate of each group; requires y.
    km_ci : bool, default=True
        Draw the 95% log-log confidence interval of the Kaplan-Meier estimates.
    n_std : float, default=1.96
        Half-width of the expected survival band in standard deviations.
    n_cols : int, default=4
        Maximum number of panel columns.

    Returns
    -------
    matplotlib.figure.Figure
        With one panel per group; with several groups, a first panel overlays
        all of them without confidence bands.

    Notes
    -----
    Given the predicted survival :math:`S_i(t)` of the :math:`n` samples of a
    group, the number of samples surviving at :math:`t` is a sum of
    independent Bernoulli variables, so the surviving fraction has mean
    :math:`\frac{1}{n}\sum_i S_i(t)` and standard deviation
    :math:`\frac{1}{n}\sqrt{\sum_i S_i(t)(1 - S_i(t))}`. The band is
    clipped to [0, 1].
    """
    X = np.asarray(X)
    partition = np.ones(len(X), dtype=int) if partition is None else np.asarray(partition)
    times, events = _extract_times_events(y)
    if times is not None:
        times, events = np.asarray(times, dtype=float), np.asarray(events, dtype=bool)
    if time_grid is None:
        max_time = times.max() if times is not None else detect_max_survival_time(model, X)
        time_grid = np.linspace(0.0, max_time, 200)
    survival = model.predict("survival", X, time_grid)
    show_km = show_km and times is not None

    labels = np.unique(partition)
    colors = _get_colors(len(labels))
    # Overview panel only when there are several groups.
    n_panels = len(labels) + (len(labels) > 1)
    n_cols = min(n_cols, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 4.2 * n_rows),
                             sharex=True, sharey=True, squeeze=False, layout="constrained")
    panels = axes.flat

    def draw(ax, label, color, overview):
        mask = partition == label
        # Fainter individual curves in large groups and when groups overlap.
        individual_alpha = np.exp(-mask.sum() / 500) / (len(labels) if overview else 1) if show_individual else None
        _draw_group_curves(
            ax, survival[mask], time_grid, color, n_std, show_ci and not overview, individual_alpha,
            times[mask] if show_km else None, events[mask] if show_km else None,
            km_color=color if overview else "0.15", km_ci=km_ci and not overview,
        )

    def counts(mask):
        return f"n={mask.sum()}" + (f", events={events[mask].sum()}" if show_km else "")

    if len(labels) > 1:
        for label, color in zip(labels, colors):
            draw(panels[0], label, color, overview=True)
        panels[0].set_title(f"All groups  ({counts(partition == partition)})", loc="left")
    for ax, label, color in zip(panels[n_panels - len(labels):], labels, colors):
        draw(ax, label, color, overview=False)
        mask = partition == label
        ax.set_title(f"Group {label}  ({counts(mask)})" if len(labels) > 1 else counts(mask), loc="left")

    for i, ax in enumerate(panels):
        if i >= n_panels:
            ax.set_visible(False)
            continue
        ax.grid(alpha=0.3)
        ax.set_xlabel("Time")
        ax.set_ylabel("Survival probability")
        # Last visible panel of its column shows the time axis.
        if i + n_cols >= n_panels:
            ax.xaxis.set_tick_params(labelbottom=True)
    axes[0, 0].set_xlim(time_grid[0], time_grid[-1])
    axes[0, 0].set_ylim(0, 1.02)

    handles = [Line2D([], [], color="0.15", linewidth=2.5, label="Expected survival"
                      + (f" (band: \u00b1 {n_std:g} SD)" if show_ci else ""))]
    if show_individual:
        handles.append(Line2D([], [], color="0.15", linewidth=0.6, label="Individual predicted survival"))
    if show_km:
        handles.append(Line2D([], [], color="0.15", linewidth=2, linestyle="--",
                              label="Kaplan-Meier" + (" (95% CI)" if km_ci else "")))
    if len(labels) > 1:
        handles += [Line2D([], [], color=color, linewidth=2.5, label=f"Group {label}")
                    for label, color in zip(labels, colors)]
    fig.legend(handles=handles, loc="outside lower left", ncol=min(len(handles), 4), frameon=False, fontsize=9)
    return fig


def get_training_history_table(model):
    r"""Convert recorded batch and test losses to a two-level table.

    Parameters
    ----------
    model : SurvivalPredictor
        Predictor with ``train_history_`` populated by ``history=True``.

    Returns
    -------
    pandas.DataFrame
        Rows are epochs; columns group train batch indices and test loss
        names. Unequal batch counts are padded with missing values.

    Notes
    -----
    The implementation expects entries to expose item(). Current predictor
    test-loss history contains Python floats, which do not expose it; review
    this helper before using it with test-loss history.
    """
    train_losses = pd.DataFrame(
        [[l.item() for l in epoch[0]] for epoch in model.train_history_]
    )
    test_losses = pd.DataFrame(
        [{n: v.item() for n, v in epoch[1].items()} for epoch in model.train_history_]
    )
    return pd.concat({"train": train_losses, "test": test_losses}, axis=1)


def plot_training_history(model):
    r"""Plot normalized epoch training and test losses on a log scale.

    Parameters
    ----------
    model : SurvivalPredictor
        Predictor with ``train_history_`` populated by ``history=True``.

    Returns
    -------
    None
        Creates a matplotlib figure.

    Notes
    -----
    Training loss is the unweighted mean of usable batch losses. Every
    series is divided by its maximum before plotting; zero maxima, zero
    values, and negative objectives require care on a log axis.
    """
    train_loss = pd.DataFrame(
        [epoch[0] for epoch in model.train_history_]
    )
    test_losses = pd.DataFrame(
        [{n: v for n, v in epoch[1].items()} for epoch in model.train_history_]
    )
    df = test_losses.copy()
    df["train_loss"] = train_loss.mean(axis=1)
    df = df / df.max()
    plt.figure(figsize=(10, 6))
    seaborn.lineplot(data=df)
    plt.yscale("log")
