#!/usr/bin/env python3
"""Construct time-resolved classical shadow matrices from measurement records.

Expected input dataframe columns:
- time
- shadow_id
- qubit
- pauli
- outcome

For each (time, shadow_id), this builds the single-shot shadow estimator:
    rho_hat = ⊗_q (3 |psi_q><psi_q| - I)
Then it averages over all shadow_id at the same time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import qutip as qt


# Pauli eigenstates used for classical-shadow inversion
ket0 = qt.basis(2, 0)
ket1 = qt.basis(2, 1)
pauli_eigen = {
    "Z": {+1: ket0, -1: ket1},
    "X": {+1: (ket0 + ket1).unit(), -1: (ket0 - ket1).unit()},
    "Y": {+1: (ket0 + 1j * ket1).unit(), -1: (ket0 - 1j * ket1).unit()},
}


def _single_qubit_shadow_operator(pauli_label: str, outcome: int) -> qt.Qobj:
    """Return 3|psi><psi| - I for one qubit measurement result."""
    if pauli_label not in pauli_eigen:
        raise ValueError(f"Unsupported pauli label: {pauli_label}")
    if outcome not in (+1, -1):
        raise ValueError(f"Outcome must be +/-1, got: {outcome}")

    ket = pauli_eigen[pauli_label][int(outcome)]
    proj = ket * ket.dag()
    return 3 * proj - qt.qeye(2)


def _shadow_from_one_shot(shot_df: pd.DataFrame, qubit_num: int) -> qt.Qobj:
    """Construct tensor-product classical shadow for one (time, shadow_id)."""
    if shot_df["qubit"].nunique() != qubit_num:
        raise ValueError(
            "Each (time, shadow_id) must contain exactly one record per qubit. "
            f"Expected {qubit_num}, got {shot_df['qubit'].nunique()}."
        )

    shot_df = shot_df.sort_values("qubit")
    ops = []
    seen = set()

    for row in shot_df.itertuples(index=False):
        q = int(row.qubit)
        if q in seen:
            raise ValueError(f"Duplicate qubit record found in one shot: qubit={q}")
        seen.add(q)

        op = _single_qubit_shadow_operator(str(row.pauli), int(row.outcome))
        ops.append(op)

    return qt.tensor(ops)


def construct_classical_shadow_matrices_by_time(
    measurement_df: pd.DataFrame,
    qubit_num: int,
) -> tuple[dict[float, qt.Qobj], np.ndarray, np.ndarray]:
    """Build classical shadow density estimates grouped by time.

    Parameters
    ----------
    measurement_df
        DataFrame with columns: time, shadow_id, qubit, pauli, outcome.
    qubit_num
        Number of qubits.

    Returns
    -------
    time_to_shadow_matrix
        Dict mapping each time value -> averaged classical shadow (Qobj).
    times
        Sorted 1D array of time points.
    shadow_matrices
        Complex ndarray of shape (num_times, 2**qubit_num, 2**qubit_num).
    """
    required_cols = {"time", "shadow_id", "qubit", "pauli", "outcome"}
    missing = required_cols.difference(measurement_df.columns)
    if missing:
        raise ValueError(f"measurement_df missing columns: {sorted(missing)}")
    if qubit_num <= 0:
        raise ValueError("qubit_num must be positive.")

    df = measurement_df.copy()
    df["time"] = df["time"].astype(float)
    df["shadow_id"] = df["shadow_id"].astype(int)
    df["qubit"] = df["qubit"].astype(int)
    df["pauli"] = df["pauli"].astype(str)
    df["outcome"] = df["outcome"].astype(int)

    shots_by_time: dict[float, list[qt.Qobj]] = {}

    grouped = df.groupby(["time", "shadow_id"], sort=True)
    for (time, _shadow_id), shot_df in grouped:
        shot_shadow = _shadow_from_one_shot(shot_df, qubit_num=qubit_num)
        shots_by_time.setdefault(float(time), []).append(shot_shadow)

    time_to_shadow_matrix: dict[float, qt.Qobj] = {}
    for time in sorted(shots_by_time.keys()):
        shot_list = shots_by_time[time]
        avg_shadow = shot_list[0] * 0
        for s in shot_list:
            avg_shadow = avg_shadow + s
        avg_shadow = avg_shadow / len(shot_list)
        time_to_shadow_matrix[time] = avg_shadow

    times = np.array(sorted(time_to_shadow_matrix.keys()), dtype=float)
    shadow_matrices = np.stack([time_to_shadow_matrix[t].full() for t in times], axis=0)

    return time_to_shadow_matrix, times, shadow_matrices


__all__ = ["construct_classical_shadow_matrices_by_time"]
