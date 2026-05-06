# Code Structure and Function Guide

This project is organized around two core modules:

- `quantum_part.py`: quantum measurement simulation, classical shadow construction, and observable estimation.
- `Bayesian_part.py`: GP-based Bayesian learning for measurement probabilities (including conditional/autoregressive modeling).

## High-level Pipeline

1. Generate measurement data from quantum dynamics (`quantum_part.py`).
2. Train Bayesian models on measurement outcomes (`Bayesian_part.py`).
3. Use learned probabilities to simulate new shadows (`quantum_part.py`).
4. Reconstruct observable dynamics via classical shadows (`quantum_part.py`).
5. Visualize learned conditional probabilities (`Bayesian_part.py`).

---

## `quantum_part.py`

### Measurement and Shadow Basics

- `create_random_pauli_obs(qubit_num)`
  - Samples random local Pauli bases (`X/Y/Z`) for each qubit.

- `pauli_measurement(rho, pauli_obs_list, qubit_num)`
  - Performs sequential local projective measurement on each qubit.
  - Uses post-measurement state update, so later qubits are physically conditioned on earlier outcomes.

- `create_classical_shadow(data)`
  - Converts one-shot measurement records into classical shadow estimators:
    - Per qubit estimator: `3|psi><psi| - I`
    - Full estimator: tensor product across qubits.

- `estimation_mean(shadow_list, operator)`
  - Computes observable estimate from shadows:
    - mean of `Tr(operator * rho_hat_shot)` over shots.

### Simulation from Learned Probabilities

- `simulate_classical_shadow(all_probs, measurement_df, qubit_num, shadow_id, t_index)`
  - Original independent sampler using `all_probs[q, s, t]`.
  - Does **not** condition qubit `q` on previous sampled outcomes.

- `simulate_classical_shadow_conditional(model_bank, measurement_df, qubit_num, shadow_id, time_value)`
  - New autoregressive sampler.
  - For qubit `q`, samples from:
    - `P(m_q = +1 | time, m_0, ..., m_{q-1}, shadow_id)`
  - Uses previously sampled outcomes in the same shot.

### Reconstructing Operator Dynamics

- `reconstruct_operator_dynamics_from_conditional_models(model_bank, measurement_df, qubit_num, operator, time_points, shadow_ids=None)`
  - For each target time:
    1. Samples one shot per shadow using conditional sampler.
    2. Builds classical shadow estimators.
    3. Estimates observable using `estimation_mean`.
  - Returns:
    - `times`: queried times
    - `estimation`: reconstructed `<operator>(t)`
    - `shots`: sampled measurement records
    - `shadow_ids`: used shadow ids
  - Performance optimizations included:
    - caches conditional GP probability curves across all times
    - precomputes Pauli lookup table
    - sets GP models to eval once
    - progress bar via `tqdm` (fallback if unavailable)

---

## `Bayesian_part.py`

### Core GP Classifier

- `GPBinaryVI`
  - Sparse variational GP for binary classification.
  - Uses constant mean + scaled Matern kernel.

- `train_gp_classifier(train_x, train_y, num_inducing=50, training_iter=200, lr=0.05)`
  - Trains variational GP classifier with Bernoulli likelihood.
  - Supports labels in `{0,1}` or `{-1,+1}`.

- `predict_gp_classifier(model, likelihood, test_x)`
  - Returns predicted class probability and latent posterior moments.

### Conditional / Autoregressive Bayesian Modeling

- `build_conditional_training_data(measurement_df, qubit, shadow_id)`
  - Builds supervised dataset for one `(qubit, shadow_id)`:
    - Inputs:
      - qubit `0`: `[time]`
      - qubit `q>0`: `[time, outcome_0, ..., outcome_{q-1}]`
    - Target:
      - outcome of qubit `q`.

- `train_conditional_gp_classifiers(measurement_df, qubit_num, shadow_ids=None, ...)`
  - Trains one GP classifier per `(qubit, shadow_id)`.
  - Output format:
    - `model_bank[q][s] = {"model": ..., "likelihood": ...}`

- `predict_conditional_gp_probability(model_entry, time_value, previous_outcomes=None)`
  - Single-point probability prediction:
    - `P(outcome=+1 | time_value, previous_outcomes)`.

- `predict_conditional_gp_curve(model_entry, t_values, previous_outcomes=None)`
  - Vectorized probability predictions across many times.

### Visualization

- `plot_conditional_probability_curves(model_bank, qubit, shadow_id, t_values, conditioning_patterns=None, measurement_df=None, ax=None)`
  - Plots learned conditional probabilities over time.
  - For `qubit=0`: single unconditional curve vs time.
  - For `qubit>0`: one curve per conditioning pattern (for example `m0=+1` vs `m0=-1`).
  - Returns `(fig, ax, curves)` where `curves` stores numeric outputs.

---

## Typical Usage Pattern

```python
from quantum_part import reconstruct_operator_dynamics_from_conditional_models
from Bayesian_part import train_conditional_gp_classifiers, plot_conditional_probability_curves

# 1) Train conditional models
model_bank = train_conditional_gp_classifiers(
    measurement_df=measurement_df,
    qubit_num=qubit_num,
    shadow_ids=range(shadow_size),
)

# 2) Reconstruct operator dynamics from learned conditional probabilities
recon_times = tlist[indices]
results = reconstruct_operator_dynamics_from_conditional_models(
    model_bank=model_bank,
    measurement_df=measurement_df,
    qubit_num=qubit_num,
    operator=operator,
    time_points=recon_times,
    shadow_ids=range(shadow_size),
)

estimation_list = results["estimation"]

# 3) Visualize one conditional probability model
fig, ax, curves = plot_conditional_probability_curves(
    model_bank=model_bank,
    qubit=1,
    shadow_id=0,
    t_values=tlist,
    measurement_df=measurement_df,
)
```

---

## Conceptual Note

The key correction is moving from independent qubit models to an autoregressive factorization:

`P(m_0, ..., m_{n-1} | t, shadow) = Π_q P(m_q | t, m_{<q}, shadow)`

This matches sequential measurement structure and preserves dependence between qubit outcomes in generated shadows.
