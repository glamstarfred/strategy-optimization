#!/usr/bin/env python3
"""Strategy optimization at fixed total measurement budget.

This module compares how the allocation

    total_shots = shadow_size * number_of_time_indices

affects two Bayesian reconstruction methods:

1. Bayesian regression on classical-shadow matrices.
2. Conditional-probability Bayesian inference.

For each allocation strategy, the module regenerates measurement data,
reconstructs the observable dynamics, and computes error against the
theoretical values from ``qutip.mesolve``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import qutip as qt
import matplotlib.pyplot as plt

from error_comparison import (
    _compute_error,
    _measurement_df_for_shadow_size,
    _run_conditional_method,
    _run_matrix_method,
)


@dataclass
class StrategyOptimizationConfig:
    """Configuration for fixed-budget strategy optimization.

    Parameters
    ----------
    total_shots
        Fixed product ``shadow_size * num_time_indices``.
    qubit_num
        Number of qubits in the simulated system.
    strategies
        Optional explicit list of ``(shadow_size, num_time_indices)`` pairs.
        If omitted, all divisor pairs compatible with ``max_time_indices`` are used.
    max_time_indices
        Maximum number of time indices allowed when automatically generating
        strategies. If ``None``, all entries in ``tlist`` are available.
    min_shadow_size, max_shadow_size
        Optional bounds for automatic strategy generation.
    training_iter, num_inducing, gp_lr
        Conditional GP training settings.
    matrix_kernel, matrix_prior_mean, matrix_prior_std, matrix_credible_mass,
    matrix_jitter
        Matrix Bayesian regression settings.
    error_metric
        One of ``{"l1", "l2", "rmse"}``. ``rmse`` is the default because it
        is comparable across strategies with different numbers of time points.
    show_progress
        Whether to show a tqdm progress bar when available.
    """

    total_shots: int
    qubit_num: int
    strategies: Sequence[tuple[int, int]] | None = None
    max_time_indices: int | None = None
    min_shadow_size: int = 1
    max_shadow_size: int | None = None
    training_iter: int = 200
    num_inducing: int = 50
    gp_lr: float = 0.05
    matrix_kernel: str = "rbf"
    matrix_prior_mean: float = 0.0
    matrix_prior_std: float = 1.0
    matrix_credible_mass: float = 0.95
    matrix_jitter: float = 1e-8
    error_metric: str = "rmse"
    show_progress: bool = True


def _validate_total_shots(total_shots: int) -> int:
    total_shots = int(total_shots)
    if total_shots <= 0:
        raise ValueError("total_shots must be a positive integer.")
    return total_shots


def _evenly_spaced_indices(pool_indices: np.ndarray, count: int) -> np.ndarray:
    """Pick ``count`` approximately evenly spaced entries from ``pool_indices``."""
    pool_indices = np.asarray(pool_indices, dtype=int).reshape(-1)
    count = int(count)
    if count <= 0:
        raise ValueError("count must be positive.")
    if count > pool_indices.size:
        raise ValueError(
            f"Cannot choose {count} time indices from pool of size {pool_indices.size}."
        )
    if count == pool_indices.size:
        return pool_indices.copy()

    positions = np.linspace(0, pool_indices.size - 1, count)
    selected_positions = np.rint(positions).astype(int)

    # Rounding can duplicate positions for small pools. Fall back to a stable
    # unique selection that preserves broad coverage of the full interval.
    selected_positions = np.unique(selected_positions)
    if selected_positions.size < count:
        missing = count - selected_positions.size
        extras = np.setdiff1d(np.arange(pool_indices.size), selected_positions)
        extra_positions = np.rint(
            np.linspace(0, extras.size - 1, missing)
        ).astype(int)
        selected_positions = np.sort(
            np.concatenate([selected_positions, extras[extra_positions]])
        )

    return pool_indices[selected_positions[:count]]


def generate_fixed_budget_strategies(
    total_shots: int,
    max_time_indices: int,
    min_shadow_size: int = 1,
    max_shadow_size: int | None = None,
) -> list[tuple[int, int]]:
    """Generate valid ``(shadow_size, num_time_indices)`` divisor pairs."""
    total_shots = _validate_total_shots(total_shots)
    max_time_indices = int(max_time_indices)
    min_shadow_size = int(min_shadow_size)
    if max_time_indices <= 0:
        raise ValueError("max_time_indices must be positive.")
    if min_shadow_size <= 0:
        raise ValueError("min_shadow_size must be positive.")
    if max_shadow_size is not None and int(max_shadow_size) <= 0:
        raise ValueError("max_shadow_size must be positive when provided.")

    pairs = []
    for shadow_size in range(min_shadow_size, total_shots + 1):
        if total_shots % shadow_size != 0:
            continue
        if max_shadow_size is not None and shadow_size > int(max_shadow_size):
            continue
        num_time_indices = total_shots // shadow_size
        if num_time_indices <= max_time_indices:
            pairs.append((int(shadow_size), int(num_time_indices)))

    if not pairs:
        raise ValueError(
            "No valid strategies found. Increase max_time_indices or adjust "
            "shadow-size bounds."
        )
    return pairs


def _validate_strategies(
    strategies: Iterable[tuple[int, int]],
    total_shots: int,
    max_time_indices: int,
) -> list[tuple[int, int]]:
    total_shots = _validate_total_shots(total_shots)
    validated = []
    for pair in strategies:
        if len(pair) != 2:
            raise ValueError("Each strategy must be (shadow_size, num_time_indices).")
        shadow_size, num_time_indices = int(pair[0]), int(pair[1])
        if shadow_size <= 0 or num_time_indices <= 0:
            raise ValueError("Strategy values must be positive integers.")
        if shadow_size * num_time_indices != total_shots:
            raise ValueError(
                f"Strategy {(shadow_size, num_time_indices)} does not preserve "
                f"total_shots={total_shots}."
            )
        if num_time_indices > max_time_indices:
            raise ValueError(
                f"Strategy {(shadow_size, num_time_indices)} asks for more time "
                f"indices than available ({max_time_indices})."
            )
        validated.append((shadow_size, num_time_indices))

    if not validated:
        raise ValueError("At least one strategy is required.")
    return sorted(validated, key=lambda x: (x[0], x[1]))


def optimize_fixed_budget_strategy(
    mesolve_result,
    tlist: np.ndarray,
    operator: qt.Qobj,
    config: StrategyOptimizationConfig,
    time_index_pool: np.ndarray | None = None,
    rng_seed: int | None = None,
) -> pd.DataFrame:
    """Evaluate fixed-budget strategies for both Bayesian methods.

    Parameters
    ----------
    mesolve_result
        Output object from ``qt.mesolve(...)``, must expose ``.states``.
    tlist
        Full time grid corresponding to ``mesolve_result.states``.
    operator
        Observable used for theoretical comparison.
    config
        Strategy optimization configuration.
    time_index_pool
        Optional pool of integer indices inside ``tlist``. Strategies choose
        evenly spaced subsets from this pool. If ``None``, the full time grid
        is used.
    rng_seed
        Optional base seed for reproducible measurement generation.

    Returns
    -------
    pandas.DataFrame
        Columns include ``shadow_size``, ``num_time_indices``,
        ``total_shots``, ``conditional_error``, and ``matrix_error``.
    """
    total_shots = _validate_total_shots(config.total_shots)
    tlist = np.asarray(tlist, dtype=float).reshape(-1)
    if tlist.size == 0:
        raise ValueError("tlist must not be empty.")
    if len(mesolve_result.states) < tlist.size:
        raise ValueError("mesolve_result.states must cover all entries in tlist.")

    if time_index_pool is None:
        time_index_pool = np.arange(tlist.size, dtype=int)
    else:
        time_index_pool = np.asarray(time_index_pool, dtype=int).reshape(-1)
        if time_index_pool.size == 0:
            raise ValueError("time_index_pool must not be empty.")
        if np.any(time_index_pool < 0) or np.any(time_index_pool >= tlist.size):
            raise ValueError("time_index_pool contains indices outside tlist.")

    max_time_indices = (
        int(config.max_time_indices)
        if config.max_time_indices is not None
        else int(time_index_pool.size)
    )
    max_time_indices = min(max_time_indices, int(time_index_pool.size))

    if config.strategies is None:
        strategies = generate_fixed_budget_strategies(
            total_shots=total_shots,
            max_time_indices=max_time_indices,
            min_shadow_size=config.min_shadow_size,
            max_shadow_size=config.max_shadow_size,
        )
    else:
        strategies = _validate_strategies(
            config.strategies,
            total_shots=total_shots,
            max_time_indices=max_time_indices,
        )

    iterator = strategies
    if config.show_progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(strategies, desc="Fixed-budget strategy sweep")
        except ImportError:
            iterator = strategies

    rows = []
    for strategy_idx, (shadow_size, num_time_indices) in enumerate(iterator):
        indices = _evenly_spaced_indices(time_index_pool, num_time_indices)
        recon_times = tlist[indices]
        true_values = np.real(
            np.asarray(
                qt.expect(operator, [mesolve_result.states[i] for i in indices])
            )
        )

        local_seed = None if rng_seed is None else int(rng_seed) + strategy_idx
        measurement_df = _measurement_df_for_shadow_size(
            states=mesolve_result.states,
            tlist=tlist,
            indices=indices,
            qubit_num=config.qubit_num,
            shadow_size=shadow_size,
            rng_seed=local_seed,
        )

        conditional_est = _run_conditional_method(
            measurement_df=measurement_df,
            recon_times=recon_times,
            operator=operator,
            qubit_num=config.qubit_num,
            shadow_size=shadow_size,
            training_iter=config.training_iter,
            num_inducing=config.num_inducing,
            gp_lr=config.gp_lr,
        )
        matrix_est = _run_matrix_method(
            measurement_df=measurement_df,
            operator=operator,
            qubit_num=config.qubit_num,
            recon_times=recon_times,
            kernel=config.matrix_kernel,
            prior_mean=config.matrix_prior_mean,
            prior_std=config.matrix_prior_std,
            credible_mass=config.matrix_credible_mass,
            jitter=config.matrix_jitter,
        )

        rows.append(
            {
                "shadow_size": int(shadow_size),
                "num_time_indices": int(num_time_indices),
                "total_shots": int(total_shots),
                "time_indices": tuple(int(i) for i in indices.tolist()),
                "conditional_error": _compute_error(
                    conditional_est, true_values, metric=config.error_metric
                ),
                "matrix_error": _compute_error(
                    matrix_est, true_values, metric=config.error_metric
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["shadow_size", "num_time_indices"])
        .reset_index(drop=True)
    )


def plot_fixed_budget_strategy_errors(
    strategy_df: pd.DataFrame,
    metric_label: str = "RMSE",
    title: str = "Fixed-Budget Strategy Optimization",
    ax=None,
):
    """Plot method errors across fixed-budget allocation strategies.

    The x-axis is shadow size. A secondary top axis shows the corresponding
    number of time indices, so each point represents one fixed-budget pair.
    """
    required = {
        "shadow_size",
        "num_time_indices",
        "conditional_error",
        "matrix_error",
    }
    missing = required.difference(strategy_df.columns)
    if missing:
        raise ValueError(f"strategy_df missing required columns: {sorted(missing)}")

    df = strategy_df.sort_values("shadow_size").reset_index(drop=True)
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4.8))
    else:
        fig = ax.figure

    x = df["shadow_size"].to_numpy(dtype=float)
    ax.plot(
        x,
        df["conditional_error"].to_numpy(dtype=float),
        marker="o",
        linewidth=2,
        label="Conditional Bayesian",
    )
    ax.plot(
        x,
        df["matrix_error"].to_numpy(dtype=float),
        marker="s",
        linewidth=2,
        label="Matrix Bayesian",
    )

    ax.set_xlabel("Shadow Size")
    ax.set_ylabel(metric_label)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()

    top = ax.secondary_xaxis("top")
    top.set_xticks(x)
    top.set_xticklabels(df["num_time_indices"].astype(str).tolist())
    top.set_xlabel("Number of Time Indices")

    fig.tight_layout()
    return fig, ax


__all__ = [
    "StrategyOptimizationConfig",
    "generate_fixed_budget_strategies",
    "optimize_fixed_budget_strategy",
    "plot_fixed_budget_strategy_errors",
]
