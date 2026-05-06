#!/usr/bin/env python3
"""Compare matrix Bayesian inference vs conditional Bayesian inference.

This module runs both methods over a user-defined list of shadow sizes,
computes cumulative error against the true observable dynamics, and plots
error versus shadow size.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import qutip as qt
import matplotlib.pyplot as plt

from quantum_part import (
    create_random_pauli_obs,
    pauli_measurement,
    reconstruct_operator_dynamics_from_conditional_models,
    estimate_observable_from_bayesian_results,
)
from Bayesian_part import train_conditional_gp_classifiers
from classical_shadow_matrix import construct_classical_shadow_matrices_by_time
from bayesian_matrix_inference_botorch import run_bayesian_matrix_inference


@dataclass
class ComparisonConfig:
    """Configuration for error comparison."""

    shadow_sizes: Iterable[int]
    qubit_num: int
    training_iter: int = 200
    num_inducing: int = 50
    gp_lr: float = 0.05
    matrix_kernel: str = "rbf"
    matrix_prior_mean: float = 0.0
    matrix_prior_std: float = 1.0
    matrix_credible_mass: float = 0.95
    matrix_jitter: float = 1e-8
    error_metric: str = "l1"  # one of {"l1", "l2", "rmse"}
    show_progress: bool = True
    cumulative_subsets: bool = True


def _validate_shadow_sizes(shadow_sizes: Iterable[int]) -> list[int]:
    sizes = [int(s) for s in shadow_sizes]
    if len(sizes) == 0:
        raise ValueError("shadow_sizes must contain at least one value.")
    if any(s <= 0 for s in sizes):
        raise ValueError("All shadow_sizes must be positive integers.")
    return sizes


def _compute_error(y_hat: np.ndarray, y_true: np.ndarray, metric: str) -> float:
    y_hat = np.asarray(y_hat, dtype=float).reshape(-1)
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    if y_hat.shape != y_true.shape:
        raise ValueError(
            f"Shape mismatch in error computation: {y_hat.shape} vs {y_true.shape}."
        )

    diff = y_hat - y_true
    if metric == "l1":
        return float(np.sum(np.abs(diff)))
    if metric == "l2":
        return float(np.sum(diff**2))
    if metric == "rmse":
        return float(np.sqrt(np.mean(diff**2)))
    raise ValueError("Unsupported error_metric. Use one of: 'l1', 'l2', 'rmse'.")


def _measurement_df_for_shadow_size(
    states: list[qt.Qobj],
    tlist: np.ndarray,
    indices: np.ndarray,
    qubit_num: int,
    shadow_size: int,
    rng_seed: int | None = None,
) -> pd.DataFrame:
    """Generate measurement dataframe with fixed random Pauli setting per shadow."""
    if rng_seed is not None:
        np.random.seed(rng_seed)

    obs_list = [create_random_pauli_obs(qubit_num) for _ in range(shadow_size)]
    records = []

    for idx in indices:
        rho = states[idx] * states[idx].dag()
        time = float(tlist[idx])
        for shadow_id in range(shadow_size):
            measurements = pauli_measurement(rho, obs_list[shadow_id], qubit_num)
            for qubit, (pauli_label, outcome) in enumerate(measurements):
                records.append(
                    {
                        "time": time,
                        "shadow_id": int(shadow_id),
                        "qubit": int(qubit),
                        "pauli": str(pauli_label),
                        "outcome": int(outcome),
                    }
                )

    return pd.DataFrame(records)


def _run_conditional_method(
    measurement_df: pd.DataFrame,
    recon_times: np.ndarray,
    operator: qt.Qobj,
    qubit_num: int,
    shadow_size: int,
    training_iter: int,
    num_inducing: int,
    gp_lr: float,
) -> np.ndarray:
    model_bank = train_conditional_gp_classifiers(
        measurement_df=measurement_df,
        qubit_num=qubit_num,
        shadow_ids=range(shadow_size),
        num_inducing=num_inducing,
        training_iter=training_iter,
        lr=gp_lr,
    )
    rec = reconstruct_operator_dynamics_from_conditional_models(
        model_bank=model_bank,
        measurement_df=measurement_df,
        qubit_num=qubit_num,
        operator=operator,
        time_points=recon_times,
        shadow_ids=range(shadow_size),
    )
    return np.real(np.asarray(rec["estimation"], dtype=float))


def _run_matrix_method(
    measurement_df: pd.DataFrame,
    operator: qt.Qobj,
    qubit_num: int,
    recon_times: np.ndarray,
    kernel: str,
    prior_mean: float,
    prior_std: float,
    credible_mass: float,
    jitter: float,
) -> np.ndarray:
    _, obs_times, shadow_matrices = construct_classical_shadow_matrices_by_time(
        measurement_df=measurement_df,
        qubit_num=qubit_num,
    )
    matrix_results = run_bayesian_matrix_inference(
        observations=shadow_matrices,
        time_index=obs_times,
        target_time_index=recon_times,
        kernel=kernel,
        prior_mean=prior_mean,
        prior_std=prior_std,
        credible_mass=credible_mass,
        jitter=jitter,
    )
    est = estimate_observable_from_bayesian_results(matrix_results, operator)
    return np.real(np.asarray(est, dtype=float))


def compare_error_across_shadow_sizes(
    mesolve_result,
    tlist: np.ndarray,
    indices: np.ndarray,
    operator: qt.Qobj,
    config: ComparisonConfig,
    rng_seed: int | None = None,
) -> pd.DataFrame:
    """Run both methods for each shadow size and compute cumulative errors.

    Parameters
    ----------
    mesolve_result
        Output object from `qt.mesolve(...)`, must expose `.states`.
    tlist
        Full time grid.
    indices
        Indices inside `tlist` used to collect measurements/reconstruct dynamics.
    operator
        Observable for evaluation.
    config
        ComparisonConfig controlling shadow sizes and model options.
    rng_seed
        Optional seed for reproducible measurement generation.

    Returns
    -------
    pd.DataFrame
        Columns:
        - shadow_size
        - conditional_error
        - matrix_error
    """
    shadow_sizes = _validate_shadow_sizes(config.shadow_sizes)
    indices = np.asarray(indices, dtype=int).reshape(-1)
    if indices.size == 0:
        raise ValueError("indices must not be empty.")

    tlist = np.asarray(tlist, dtype=float).reshape(-1)
    recon_times = tlist[indices]
    true_values = np.real(np.asarray(qt.expect(operator, [mesolve_result.states[i] for i in indices])))

    iterator = shadow_sizes
    if config.show_progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(shadow_sizes, desc="Shadow-size sweep")
        except ImportError:
            iterator = shadow_sizes

    # Build one maximum-size dataset and use cumulative subsets for fair comparison.
    # This ensures "shadow size = s" means first s shadows from the same pool.
    max_shadow_size = int(max(shadow_sizes))
    base_seed = None if rng_seed is None else int(rng_seed)
    full_measurement_df = None
    if config.cumulative_subsets:
        full_measurement_df = _measurement_df_for_shadow_size(
            states=mesolve_result.states,
            tlist=tlist,
            indices=indices,
            qubit_num=config.qubit_num,
            shadow_size=max_shadow_size,
            rng_seed=base_seed,
        )

    rows = []
    for i, shadow_size in enumerate(iterator):
        if config.cumulative_subsets:
            measurement_df = full_measurement_df.loc[
                full_measurement_df["shadow_id"] < int(shadow_size)
            ].copy()
        else:
            local_seed = None if rng_seed is None else (int(rng_seed) + i)
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
                "conditional_error": _compute_error(
                    conditional_est, true_values, metric=config.error_metric
                ),
                "matrix_error": _compute_error(matrix_est, true_values, metric=config.error_metric),
            }
        )

    return pd.DataFrame(rows).sort_values("shadow_size").reset_index(drop=True)


def plot_error_growth_with_shadow_size(
    error_df: pd.DataFrame,
    metric_label: str = "Cumulative Error",
    title: str = "Error vs Shadow Size",
    ax=None,
):
    """Plot error curves for conditional and matrix Bayesian methods."""
    required = {"shadow_size", "conditional_error", "matrix_error"}
    missing = required.difference(error_df.columns)
    if missing:
        raise ValueError(f"error_df missing required columns: {sorted(missing)}")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
    else:
        fig = ax.figure

    df = error_df.sort_values("shadow_size")
    x = df["shadow_size"].to_numpy(dtype=float)
    y_cond = df["conditional_error"].to_numpy(dtype=float)
    y_mat = df["matrix_error"].to_numpy(dtype=float)

    ax.plot(x, y_cond, marker="o", linewidth=2, label="Conditional Bayesian")
    ax.plot(x, y_mat, marker="s", linewidth=2, label="Matrix Bayesian (BoTorch)")
    ax.set_xlabel("Shadow Size")
    ax.set_ylabel(metric_label)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax


__all__ = [
    "ComparisonConfig",
    "compare_error_across_shadow_sizes",
    "plot_error_growth_with_shadow_size",
]
