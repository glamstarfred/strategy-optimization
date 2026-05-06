# -*- coding: utf-8 -*-
try:
    import qutip as qt
except ImportError as e:
    raise ImportError(
        "The 'qutip' package is required for quantum_part.py. "
        "Install it via `pip install qutip` or conda before running."
    ) from e

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import scipy
from scipy.stats import norm

############################################################################################
# definition for all the basis and pauli matrix
paulis = {
    "X": qt.sigmax(),
    "Y": qt.sigmay(),
    "Z": qt.sigmaz()
}

ket0 = qt.basis(2, 0)
ket1 = qt.basis(2, 1)

pauli_eigen = {
    'Z': {+1: ket0, -1: ket1},
    'X': {+1: (ket0 + ket1).unit(),
          -1: (ket0 - ket1).unit()},
    'Y': {+1: (ket0 + 1j*ket1).unit(),
          -1: (ket0 - 1j*ket1).unit()}
}

############################################################################################

def random_pauli():
    """
    Randomly sample a single-qubit Pauli observable.
    
    The function selects uniformly at random from the keys of the global
    dictionary `paulis`, which is assumed to map Pauli labels
    (e.g. 'X', 'Y', 'Z') to their corresponding QuTiP operators.
    
    Returns
    -------
    label : str
        Pauli label (e.g., 'X', 'Y', 'Z').
    
    operator : qutip.Qobj
        The corresponding 2×2 Pauli operator.
    """
    key = np.random.choice(list(paulis.keys()))
    return key, paulis[key]

def projector_on_qubit(eigvec, qubit_index, qubit_num):
    """
    Construct a projector acting nontrivially on a single qubit.
    
    Given a single-qubit eigenvector |v⟩, this function builds the
    many-body operator:
    
        I ⊗ ... ⊗ |v⟩⟨v| ⊗ ... ⊗ I
    
    where the projector |v⟩⟨v| is placed at position `qubit_index`
    in an `qubit_num`-qubit Hilbert space.
    
    Parameters
    ----------
    eigvec : qutip.Qobj
        Single-qubit eigenvector (dimension 2×1).
    
    qubit_index : int
        Target qubit index (0-based).
    
    qubit_num : int
        Total number of qubits.
    
    Returns
    -------
    projector : qutip.Qobj
        Tensor-product projector acting on the full Hilbert space
        (dimension 2^n × 2^n).
    """
    ops = []
    for i in range(qubit_num):
        if i == qubit_index:
            ops.append(eigvec * eigvec.dag())
        else:
            ops.append(qt.qeye(2))
    return qt.tensor(ops)
    

            
def create_classical_shadow(data):
    """
    Construct classical shadow estimators from measurement data.

    For each measurement shot consisting of local Pauli outcomes,
    this function constructs the single-shot classical shadow estimator:

        ρ̂ = ⊗_i (3 |ψ_i⟩⟨ψ_i| − I)

    where |ψ_i⟩ is the measured eigenstate corresponding to the
    observed Pauli outcome on qubit i.

    Parameters
    ----------
    data : list
        List of measurement shots.
        Each shot is a list of (pauli_label, outcome) pairs.

    Returns
    -------
    shadow_list : list of qutip.Qobj
        List of tensor-product shadow estimators (dimension 2^n × 2^n),
        one per shot.
    """
    shadow_list = []
    for shot in data:
        rhohat_single_shot = []
        for pauli, outcome in shot:
            ket = pauli_eigen[pauli][outcome]
            proj = ket * ket.dag()
            rhohat_single_shot.append(3 * proj - qt.qeye(2))

        shadow_list.append(qt.tensor(rhohat_single_shot))

    return shadow_list

def estimation_mean(shadow_list, operator):
    """
    Estimate the expectation value of an operator via sample mean.

    Computes:
        ⟨O⟩ ≈ (1/N) Σ_i Tr(O ρ̂_i)

    Parameters
    ----------
    shadow_list : list of qutip.Qobj
        List of classical shadow estimators.

    operator : qutip.Qobj
        Observable whose expectation value is to be estimated.

    Returns
    -------
    float
        Sample mean estimator of the expectation value.
    """
    values = np.array([qt.expect(operator, s) for s in shadow_list], dtype=float)
    return np.mean(values)



def create_random_pauli_obs(qubit_num):
    """
    Generate a list of randomly chosen Pauli observables.

    Parameters
    ----------
    qubit_num : int
        Number of qubits.

    Returns
    -------
    pauli_obs_list : list of str
        List of length `qubit_num` containing Pauli labels
        ('X', 'Y', 'Z') sampled uniformly at random.
    """
    pauli_obs_list=[]
    for q in range(qubit_num):
        label, op = random_pauli()
        pauli_obs_list.append(label)
    return pauli_obs_list

def pauli_measurement(rho, pauli_obs_list, qubit_num):
    """
    Perform a local Pauli measurement with fixed measurement bases.

    Unlike `shadow_measurement`, the Pauli observables are provided
    externally instead of being sampled randomly.

    For each qubit i:
        1. Use `pauli_obs_list[i]` as the measurement basis.
        2. Compute Born-rule probabilities for ±1 outcomes.
        3. Sample outcome accordingly.

    Parameters
    ----------
    rho : qutip.Qobj
        n-qubit density matrix.

    pauli_obs_list : list of str
        List of Pauli labels ('X', 'Y', 'Z'),
        one per qubit.

    qubit_num : int
        Number of qubits.

    Returns
    -------
    measurement_list : list of (str, int)
        List of (pauli_label, outcome) pairs.
    """
    measurement_list = []
    rho_current = rho
    for i,pauli_label in enumerate(pauli_obs_list):
        op = paulis[pauli_label]
        eigvals, eigvecs = op.eigenstates()

        proj = {}
        for val, vec in zip(eigvals, eigvecs):
            proj[int(np.sign(val))] = projector_on_qubit(vec, i, qubit_num)

        probs = [
            qt.expect(proj[+1], rho_current),
            qt.expect(proj[-1], rho_current)
        ]

        probs = np.real(probs)
        probs = np.clip(probs, 0.0, 1.0)
        probs /= probs.sum()

        outcome = np.random.choice([+1, -1], p=probs)
        measurement_list.append((pauli_label, outcome))

        # Project and renormalize the post-measurement state before next qubit.
        p_outcome = probs[0] if outcome == +1 else probs[1]
        if p_outcome <= 0:
            raise ValueError(
                f"Encountered zero probability for sampled outcome {outcome} "
                f"on qubit {i} with basis {pauli_label}."
            )
        p_op = proj[outcome]
        rho_current = (p_op * rho_current * p_op) / p_outcome

    return measurement_list

# In[]
######################## Currently not in use ############################
def estimation_median_of_means(shadow_list, operator, shots, group_number):
    """
    Estimate expectation value using the Median-of-Means (MoM) estimator.

    The data is divided into `group_number` equal groups.
    The mean is computed within each group.
    The final estimator is the median of these group means.

    Parameters
    ----------
    shadow_list : list of qutip.Qobj
        List of classical shadow estimators.

    operator : qutip.Qobj
        Observable to estimate.

    shots : int
        Total number of shadow samples used.

    group_number : int
        Number of groups for the MoM procedure.
        Must divide `shots`.

    Returns
    -------
    float
        Median-of-means estimator.
    """

    if shots % group_number != 0:
        raise ValueError("shots must be divisible by group_number")

    group_size = shots // group_number

    # Compute per-shot estimates
    values = np.array([
        qt.expect(operator, shadow_list[s])
        for s in range(shots)
    ], dtype=float)

    # Split into groups
    group_means = []
    for g in range(group_number):
        start = g * group_size
        end = (g + 1) * group_size
        group_means.append(np.mean(values[start:end]))

    return np.median(group_means)

def random_pauli_measurement(rho, qubit_num):
    """
    Perform one round of local random Pauli measurements.

    For each qubit:
        1. A Pauli observable (X, Y, or Z) is sampled uniformly.
        2. The Born-rule probabilities for ±1 outcomes are computed.
        3. A measurement outcome is sampled accordingly.

    Parameters
    ----------
    rho : qutip.Qobj
        Density matrix of the n-qubit quantum state.

    qubit_num : int
        Number of qubits in the system.

    Returns
    -------
    measurement_list : list of (str, int)
        List of length `qubit_num`, where each element is:
            (pauli_label, outcome)
        with outcome ∈ {+1, -1}.
    """
    measurement_list = []

    for q in range(qubit_num):
        label, op = random_pauli()
        eigvals, eigvecs = op.eigenstates()

        proj = {}
        for val, vec in zip(eigvals, eigvecs):
            proj[int(np.sign(val))] = projector_on_qubit(vec, q, qubit_num)

        probs = [
            qt.expect(proj[+1], rho),
            qt.expect(proj[-1], rho)
        ]

        probs = np.real(probs)
        probs = np.clip(probs, 0.0, 1.0)
        probs /= probs.sum()

        outcome = np.random.choice([+1, -1], p=probs)
        measurement_list.append((label, outcome))

    return measurement_list

def simulate_classical_shadow(all_probs, measurement_df, qubit_num, shadow_id, t_index):
    """
    Generate a single classical shadow snapshot at a specified time index
    using predicted measurement outcome probabilities.

    This function simulates one round of local Pauli measurements for an
    n-qubit system. For each qubit, it:

        1. Retrieves the Pauli observable (X, Y, or Z) assigned to the given
           shadow_id from `measurement_df`.
        2. Reads the predicted probability p = P(+1) from `all_probs`.
        3. Samples a measurement outcome from a Bernoulli distribution:
              +1 with probability p,
              -1 with probability 1 - p.
        4. Stores the (pauli_label, outcome) pair.

    Parameters
    ----------
    all_probs : torch.Tensor or np.ndarray
        Array of predicted probabilities with shape:
            (qubit_num, num_shadows, num_time_points)
        where:
            all_probs[q, s, t] = P(outcome = +1 | qubit=q,
                                                shadow=s,
                                                time=t)

    measurement_df : pandas.DataFrame
        DataFrame containing the Pauli measurement settings.
        Required columns:
            - "qubit"     : int
            - "shadow_id" : int
            - "pauli"     : str ('X', 'Y', or 'Z')
        Each (qubit, shadow_id) pair must appear exactly once.

    qubit_num : int
        Total number of qubits in the system.

    shadow_id : int
        Index specifying which classical shadow configuration to use.

    t_index : int
        Time index selecting which probability slice to sample from.

    Returns
    -------
    shadow : list of [str, int]
        A list of length `qubit_num`, where each element is:
            [pauli_label, outcome]
        with outcome from {+1, -1}.
        This represents one classical shadow measurement snapshot.

    """
    shadow = []

    for q in range(qubit_num):
        # get the Pauli observable for this qubit/shadow
        obs_row = measurement_df[
            (measurement_df["qubit"] == q) &
            (measurement_df["shadow_id"] == shadow_id)
        ]
        if obs_row.empty:
            raise ValueError(f"No observable found for qubit {q}, shadow {shadow_id}")

        pauli_label = obs_row['pauli'].values[0]  # e.g., 'X', 'Y', 'Z'

        # get predicted probability for +1 outcome at t_index
        p = all_probs[q, shadow_id, t_index].item()
        # print(p)

        # simulate outcome: +1 with prob p, -1 with prob 1-p
        outcome = 1 if random.random() < p else -1

        shadow.append([pauli_label, outcome])

    return shadow


def simulate_classical_shadow_conditional(model_bank, measurement_df, qubit_num, shadow_id, time_value):
    """
    Generate one classical-shadow snapshot using conditional autoregressive models.

    For qubit q, this samples:
        outcome_q ~ P(outcome_q | time_value, outcome_0, ..., outcome_{q-1}, shadow_id)

    where probabilities are provided by `model_bank` trained via
    `train_conditional_gp_classifiers(...)` in Bayesian_part.py.
    """
    # Local import to avoid hard dependency during non-Bayesian workflows.
    from Bayesian_part import predict_conditional_gp_probability

    shadow = []
    previous_outcomes = []

    for q in range(qubit_num):
        obs_row = measurement_df[
            (measurement_df["qubit"] == q) & (measurement_df["shadow_id"] == shadow_id)
        ]
        if obs_row.empty:
            raise ValueError(f"No observable found for qubit {q}, shadow {shadow_id}")

        pauli_label = obs_row["pauli"].values[0]

        if q not in model_bank or shadow_id not in model_bank[q]:
            raise ValueError(
                f"Missing conditional model for qubit={q}, shadow_id={shadow_id}."
            )

        p_plus = predict_conditional_gp_probability(
            model_bank[q][shadow_id],
            time_value=time_value,
            previous_outcomes=previous_outcomes,
        )

        outcome = 1 if random.random() < p_plus else -1
        previous_outcomes.append(outcome)
        shadow.append([pauli_label, outcome])

    return shadow


def reconstruct_operator_dynamics_from_conditional_models(
    model_bank,
    measurement_df,
    qubit_num,
    operator,
    time_points,
    shadow_ids=None,
):
    """
    Reconstruct operator dynamics from conditional Bayesian shadow models.

    For each time t in `time_points`:
      1. Sample one measurement shot per shadow_id using
         `simulate_classical_shadow_conditional(...)`.
      2. Convert all sampled shots into classical-shadow estimators.
      3. Estimate <operator>(t) via sample mean:
             (1/N) * sum_s Tr(operator * rhohat_s).

    Parameters
    ----------
    model_bank : dict
        Output of train_conditional_gp_classifiers(...).

    measurement_df : pandas.DataFrame
        Measurement settings and training outcomes with at least columns:
        ["time", "shadow_id", "qubit", "pauli", "outcome"].

    qubit_num : int
        Number of qubits.

    operator : qutip.Qobj
        Observable to estimate.

    time_points : array-like
        Times where dynamics should be reconstructed.

    shadow_ids : iterable or None
        Shadow ids to use. If None, inferred from measurement_df.

    Returns
    -------
    results : dict
        {
            "times": np.ndarray,                 # shape (T,)
            "estimation": np.ndarray,            # shape (T,)
            "shots": list[list[list]],           # shots[t_idx][s_idx] -> [pauli, outcome]
            "shadow_ids": list[int],
        }
    """
    # Local import to avoid hard dependency for workflows not using Bayesian models.
    from Bayesian_part import predict_conditional_gp_curve
    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None

    if shadow_ids is None:
        shadow_ids = sorted(measurement_df["shadow_id"].unique().tolist())
    else:
        shadow_ids = list(shadow_ids)

    times = np.asarray(time_points, dtype=float).reshape(-1)
    n_times = len(times)
    n_shadows = len(shadow_ids)

    # Build a fast lookup table for Pauli settings to avoid DataFrame filtering
    # in the inner loops.
    pauli_table = (
        measurement_df[["shadow_id", "qubit", "pauli"]]
        .drop_duplicates(subset=["shadow_id", "qubit"])
        .pivot(index="shadow_id", columns="qubit", values="pauli")
    )
    for s_id in shadow_ids:
        if s_id not in pauli_table.index:
            raise ValueError(f"Missing Pauli settings for shadow_id={s_id}.")
    for q in range(qubit_num):
        if q not in pauli_table.columns:
            raise ValueError(f"Missing Pauli settings for qubit={q}.")

    # Ensure all GP modules are in eval mode once (instead of per prediction call).
    for q in range(qubit_num):
        for s_id in shadow_ids:
            if q not in model_bank or s_id not in model_bank[q]:
                raise ValueError(
                    f"Missing conditional model for qubit={q}, shadow_id={s_id}."
                )
            model_bank[q][s_id]["model"].eval()
            model_bank[q][s_id]["likelihood"].eval()

    shot_at_times = [[None for _ in range(n_shadows)] for _ in range(n_times)]
    estimation = np.zeros(n_times, dtype=float)

    # Cache full time-curves for each conditioning pattern:
    # key = (q, shadow_id, prev_outcomes_tuple), value = np.ndarray shape (n_times,)
    prob_curve_cache = {}

    time_iter = range(n_times)
    if tqdm is not None:
        time_iter = tqdm(time_iter, desc="Reconstructing dynamics", leave=False)

    for t_idx in time_iter:
        shots_this_t = []

        for s_pos, s_id in enumerate(shadow_ids):
            previous_outcomes = []
            shot_data = []

            for q in range(qubit_num):
                pattern = tuple(previous_outcomes)
                cache_key = (q, s_id, pattern)

                if cache_key not in prob_curve_cache:
                    prob_curve_cache[cache_key] = predict_conditional_gp_curve(
                        model_bank[q][s_id],
                        t_values=times,
                        previous_outcomes=list(pattern),
                    )

                p_plus = float(prob_curve_cache[cache_key][t_idx])
                outcome = 1 if random.random() < p_plus else -1
                previous_outcomes.append(outcome)

                pauli_label = pauli_table.loc[s_id, q]
                shot_data.append([pauli_label, outcome])

            shot_at_times[t_idx][s_pos] = shot_data
            shots_this_t.append(shot_data)

        shadow_estimators = create_classical_shadow(shots_this_t)
        estimation[t_idx] = estimation_mean(shadow_estimators, operator)

    return {
        "times": times,
        "estimation": estimation,
        "shots": shot_at_times,
        "shadow_ids": shadow_ids,
    }

def estimate_observable_from_bayesian_results(results_botorch, operator):
    """
    Estimate <operator> from Bayesian-inferred density matrices.

    Parameters
    ----------
    results_botorch : dict
        Output dictionary from run_bayesian_matrix_inference(...), containing
        key 'posterior_mean' as shape (t, d, d) or (d, d).

    operator : qutip.Qobj
        Observable O used to compute Tr(O rho_t).

    Returns
    -------
    np.ndarray
        Complex-valued expectation array with shape (t,).
    """
    if "posterior_mean" not in results_botorch:
        raise ValueError("results_botorch must contain key 'posterior_mean'.")

    posterior = np.asarray(results_botorch["posterior_mean"])
    if posterior.ndim == 2:
        posterior = posterior[np.newaxis, :, :]
    elif posterior.ndim != 3:
        raise ValueError("posterior_mean must have shape (d,d) or (t,d,d).")

    d1, d2 = posterior.shape[1], posterior.shape[2]
    if d1 != d2:
        raise ValueError("Each inferred matrix must be square.")
    if operator.shape != (d1, d2):
        raise ValueError(
            f"Operator shape {operator.shape} does not match inferred matrix shape {(d1, d2)}."
        )

    values = []
    for k in range(posterior.shape[0]):
        rho_k = qt.Qobj(posterior[k], dims=operator.dims)
        values.append(qt.expect(operator, rho_k))

    return np.asarray(values, dtype=complex)


def plot_observable_estimation_vs_theory(
    results_botorch,
    operator,
    mesolve_result,
    estimate_times=None,
    theory_times=None,
    title="Observable Estimation vs Theory",
):
    """
    Plot Bayesian estimated observable against theoretical value (real part only).

    Parameters
    ----------
    results_botorch : dict
        Bayesian inference output dict (must contain 'posterior_mean').

    operator : qutip.Qobj
        Observable O.

    mesolve_result : qutip.solver.Result
        Output of qt.mesolve(...) containing theoretical states.

    estimate_times : array-like or None
        Time index for estimated curve. If None, uses integer indices.

    theory_times : array-like or None
        Time index for theoretical curve. If None, uses integer indices.

    title : str
        Figure title.

    Returns
    -------
    fig, ax, estimated_values, theoretical_values
    """
    estimated_values = estimate_observable_from_bayesian_results(results_botorch, operator)
    theoretical_values = np.asarray(qt.expect(operator, mesolve_result.states), dtype=complex)

    if estimate_times is None:
        x_est = np.arange(estimated_values.shape[0], dtype=float)
    else:
        x_est = np.asarray(estimate_times, dtype=float).reshape(-1)
        if x_est.shape[0] != estimated_values.shape[0]:
            raise ValueError("estimate_times length must match estimated values length.")

    if theory_times is None:
        x_theory = np.arange(theoretical_values.shape[0], dtype=float)
    else:
        x_theory = np.asarray(theory_times, dtype=float).reshape(-1)
        if x_theory.shape[0] != theoretical_values.shape[0]:
            raise ValueError("theory_times length must match theoretical values length.")

    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    ax.plot(
        x_theory,
        np.real(theoretical_values),
        linestyle="--",
        linewidth=2,
        label="Theory",
    )
    ax.plot(
        x_est,
        np.real(estimated_values),
        linestyle="-",
        linewidth=2,
        label="Bayesian",
    )
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel(r"$\langle O \rangle$")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax, estimated_values, theoretical_values


def plot_direct_bayesian_observable_vs_theory(
    results_direct_bayesian,
    operator,
    mesolve_result,
    estimate_times=None,
    theory_times=None,
    credible_mass=0.68,
    title="Direct Bayesian Observable vs Theory",
):
    """
    Plot direct Bayesian observable inference against theoretical value.

    Expected result dictionary format is the output of
    infer_observable_from_shadow_with_botorch(...), containing:
    - posterior_mean
    - optional ci_lower / ci_upper

    The plotted curves use real parts to match existing visualization style.
    """
    if "posterior_mean" not in results_direct_bayesian:
        raise ValueError("results_direct_bayesian must contain key 'posterior_mean'.")

    posterior_mean = np.asarray(results_direct_bayesian["posterior_mean"])
    if posterior_mean.ndim == 0:
        posterior_mean = posterior_mean.reshape(1)
    elif posterior_mean.ndim != 1:
        raise ValueError("posterior_mean must be scalar or 1D for observable plotting.")

    ci_lower = results_direct_bayesian.get("ci_lower", None)
    ci_upper = results_direct_bayesian.get("ci_upper", None)
    if ci_lower is not None and ci_upper is not None:
        ci_lower = np.asarray(ci_lower)
        ci_upper = np.asarray(ci_upper)
        if ci_lower.ndim == 0:
            ci_lower = ci_lower.reshape(1)
        if ci_upper.ndim == 0:
            ci_upper = ci_upper.reshape(1)
        if ci_lower.shape != posterior_mean.shape or ci_upper.shape != posterior_mean.shape:
            raise ValueError(
                "ci_lower and ci_upper must match posterior_mean shape for plotting."
            )
    else:
        ci_lower = None
        ci_upper = None

    if isinstance(operator, qt.Qobj):
        op_qobj = operator
    else:
        op_np = np.asarray(operator)
        if op_np.ndim != 2 or op_np.shape[0] != op_np.shape[1]:
            raise ValueError("operator must be square with shape (d, d).")
        dims = mesolve_result.states[0].dims if len(mesolve_result.states) > 0 else None
        op_qobj = qt.Qobj(op_np, dims=dims)

    theoretical_values = np.asarray(qt.expect(op_qobj, mesolve_result.states), dtype=complex)

    if estimate_times is None:
        x_est = np.arange(posterior_mean.shape[0], dtype=float)
    else:
        x_est = np.asarray(estimate_times, dtype=float).reshape(-1)
        if x_est.shape[0] != posterior_mean.shape[0]:
            raise ValueError("estimate_times length must match posterior_mean length.")

    if theory_times is None:
        x_theory = np.arange(theoretical_values.shape[0], dtype=float)
    else:
        x_theory = np.asarray(theory_times, dtype=float).reshape(-1)
        if x_theory.shape[0] != theoretical_values.shape[0]:
            raise ValueError("theory_times length must match theoretical values length.")

    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    ax.plot(
        x_theory,
        np.real(theoretical_values),
        linestyle="--",
        linewidth=2,
        label="Theory",
    )
    ax.plot(
        x_est,
        np.real(posterior_mean),
        linestyle="-",
        linewidth=2,
        label="Direct Bayesian",
    )
    if ci_lower is not None and ci_upper is not None:
        ax.fill_between(
            x_est,
            np.real(ci_lower),
            np.real(ci_upper),
            alpha=0.25,
            label=f"{int(round(credible_mass * 100))}% credible interval",
        )

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel(r"$\langle O \rangle$")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax, posterior_mean, ci_lower, ci_upper, theoretical_values


def estimate_observable_with_trust_zone(
    results_botorch,
    operator,
    credible_mass=0.95,
    num_samples=1000,
    seed=None,
    use_real_part=True,
):
    """
    Estimate observable and trust zone via posterior sampling.

    This uses an element-wise Gaussian approximation from:
    - results_botorch["posterior_mean"]
    - results_botorch["posterior_variance"]

    Parameters
    ----------
    results_botorch : dict
        Bayesian output dict with 'posterior_mean' and 'posterior_variance'.
    operator : qutip.Qobj
        Observable O.
    credible_mass : float
        Trust-zone mass, e.g. 0.95.
    num_samples : int
        Number of Monte Carlo posterior samples per time point.
    seed : int or None
        RNG seed.
    use_real_part : bool
        If True, build trust zone for Re(<O>); else for complex <O>.

    Returns
    -------
    mean_values : np.ndarray
        Posterior-mean observable estimate (real if use_real_part else complex).
    lower : np.ndarray
        Lower bound of trust zone.
    upper : np.ndarray
        Upper bound of trust zone.
    """
    if "posterior_mean" not in results_botorch or "posterior_variance" not in results_botorch:
        raise ValueError(
            "results_botorch must contain 'posterior_mean' and 'posterior_variance'."
        )
    if not 0 < credible_mass < 1:
        raise ValueError("credible_mass must be in (0, 1).")
    if num_samples <= 1:
        raise ValueError("num_samples must be > 1.")

    mean_mats = np.asarray(results_botorch["posterior_mean"])
    var_mats = np.asarray(results_botorch["posterior_variance"])

    if mean_mats.ndim == 2:
        mean_mats = mean_mats[np.newaxis, :, :]
    if var_mats.ndim == 2:
        var_mats = var_mats[np.newaxis, :, :]
    if mean_mats.shape != var_mats.shape:
        raise ValueError("posterior_mean and posterior_variance must have the same shape.")

    t_count, d1, d2 = mean_mats.shape
    if d1 != d2:
        raise ValueError("Inferred density matrices must be square.")
    if operator.shape != (d1, d2):
        raise ValueError(
            f"Operator shape {operator.shape} does not match inferred matrix shape {(d1, d2)}."
        )

    rng = np.random.default_rng(seed)
    alpha = (1.0 - credible_mass) / 2.0

    mean_values = np.zeros(t_count, dtype=float if use_real_part else complex)
    lower = np.zeros(t_count, dtype=float if use_real_part else complex)
    upper = np.zeros(t_count, dtype=float if use_real_part else complex)

    for t in range(t_count):
        mean_t = mean_mats[t]
        var_t = np.clip(np.real(var_mats[t]), 0.0, None)
        std_t = np.sqrt(var_t)

        if np.iscomplexobj(mean_t):
            # Split total variance equally across real/imag as a practical approximation.
            std_half = std_t / np.sqrt(2.0)
            eps_r = rng.normal(loc=0.0, scale=1.0, size=(num_samples, d1, d2))
            eps_i = rng.normal(loc=0.0, scale=1.0, size=(num_samples, d1, d2))
            rho_samples = mean_t[None, :, :] + std_half[None, :, :] * (eps_r + 1j * eps_i)
        else:
            eps = rng.normal(loc=0.0, scale=1.0, size=(num_samples, d1, d2))
            rho_samples = mean_t[None, :, :] + std_t[None, :, :] * eps

        obs_samples = []
        for s in range(num_samples):
            rho_qobj = qt.Qobj(rho_samples[s], dims=operator.dims)
            obs_samples.append(qt.expect(operator, rho_qobj))
        obs_samples = np.asarray(obs_samples, dtype=complex)

        if use_real_part:
            obs_real = np.real(obs_samples)
            mean_values[t] = np.real(np.mean(obs_samples))
            lower[t] = np.quantile(obs_real, alpha)
            upper[t] = np.quantile(obs_real, 1.0 - alpha)
        else:
            mean_values[t] = np.mean(obs_samples)
            lower[t] = np.quantile(obs_samples, alpha)
            upper[t] = np.quantile(obs_samples, 1.0 - alpha)

    return mean_values, lower, upper


def plot_observable_estimation_with_trust_zone_vs_theory(
    results_botorch,
    operator,
    mesolve_result,
    estimate_times=None,
    theory_times=None,
    credible_mass=0.95,
    num_samples=1000,
    seed=None,
    title="Observable Estimation vs Theory (with Trust Zone)",
):
    """
    Plot real-part Bayesian estimation + trust zone against theory.
    """
    est_mean, est_low, est_up = estimate_observable_with_trust_zone(
        results_botorch=results_botorch,
        operator=operator,
        credible_mass=credible_mass,
        num_samples=num_samples,
        seed=seed,
        use_real_part=True,
    )
    theoretical_values = np.asarray(qt.expect(operator, mesolve_result.states), dtype=complex)

    if estimate_times is None:
        x_est = np.arange(est_mean.shape[0], dtype=float)
    else:
        x_est = np.asarray(estimate_times, dtype=float).reshape(-1)
        if x_est.shape[0] != est_mean.shape[0]:
            raise ValueError("estimate_times length must match estimated values length.")

    if theory_times is None:
        x_theory = np.arange(theoretical_values.shape[0], dtype=float)
    else:
        x_theory = np.asarray(theory_times, dtype=float).reshape(-1)
        if x_theory.shape[0] != theoretical_values.shape[0]:
            raise ValueError("theory_times length must match theoretical values length.")

    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    ax.plot(
        x_theory,
        np.real(theoretical_values),
        linestyle="--",
        linewidth=2,
        label="Theory",
    )
    ax.plot(
        x_est,
        est_mean,
        linestyle="-",
        linewidth=2,
        label="Bayesian",
    )
    ax.fill_between(
        x_est,
        est_low,
        est_up,
        alpha=0.25,
        label=f"{int(round(credible_mass * 100))}% trust zone",
    )
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel(r"$\langle O \rangle$")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax, est_mean, est_low, est_up, theoretical_values
