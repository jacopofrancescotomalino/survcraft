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
