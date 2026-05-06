#!/usr/bin/env python3
"""BoTorch-based Bayesian inference for time-indexed matrix elements.

This module mirrors the callable interface from bayesian_matrix_inference.py,
but performs inference with BoTorch/GPyTorch Gaussian Process models.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

try:
    import torch
    import gpytorch
    from gpytorch.kernels import Kernel
    from gpytorch.kernels import RBFKernel, MaternKernel, LinearKernel, ScaleKernel
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
except ImportError as e:
    raise ImportError(
        "The 'botorch', 'gpytorch', and 'torch' packages are required for "
        "bayesian_matrix_inference_botorch.py. Install via: "
        "pip install botorch gpytorch torch"
    ) from e

KernelType = str | Kernel | Callable[[np.ndarray, np.ndarray, float, float], np.ndarray]


def _validate_square_matrices(observations: np.ndarray) -> np.ndarray:
    """Normalize observations to shape (m, n, n) and validate square matrices."""
    obs = np.asarray(observations)

    if obs.ndim == 2:
        n, d = obs.shape
        if n != d:
            raise ValueError(f"Input matrix must be square, got {n}x{d}.")
        return obs[np.newaxis, :, :]

    if obs.ndim == 3:
        m, n, d = obs.shape
        if n != d:
            raise ValueError(
                f"Each observed matrix must be square, got shape ({m}, {n}, {d})."
            )
        return obs

    raise ValueError("observations must have shape (n,n) or (m,n,n).")


def _validate_time_index(time_index: np.ndarray, m: int) -> np.ndarray:
    """Validate and return 1D time index with length m."""
    t = np.asarray(time_index, dtype=float).reshape(-1)
    if t.shape[0] != m:
        raise ValueError(
            f"time_index length must match number of observations ({m}), got {t.shape[0]}."
        )
    return t


class _CallableKernel(Kernel):
    """GPyTorch kernel wrapper for user-provided callable kernels.

    Callable signature must be:
        fn(x: np.ndarray, y: np.ndarray, length_scale: float, signal_variance: float)
    returning a 2D covariance matrix.
    """

    has_lengthscale = False

    def __init__(
        self,
        fn: Callable[[np.ndarray, np.ndarray, float, float], np.ndarray],
        length_scale: float,
        signal_variance: float,
    ) -> None:
        super().__init__()
        self._fn = fn
        self._length_scale = float(length_scale)
        self._signal_variance = float(signal_variance)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, **_) -> torch.Tensor:
        a = x1.detach().cpu().numpy().reshape(-1)
        b = x2.detach().cpu().numpy().reshape(-1)
        k = np.asarray(self._fn(a, b, self._length_scale, self._signal_variance), dtype=float)
        if k.ndim != 2:
            raise ValueError("Custom kernel callable must return a 2D covariance matrix.")
        out = torch.as_tensor(k, dtype=x1.dtype, device=x1.device)
        if diag:
            return torch.diagonal(out, dim1=-2, dim2=-1)
        return out


def _build_covariance_module(
    kernel: KernelType,
    length_scale: float,
    signal_variance: float,
    dtype: torch.dtype,
    device: torch.device,
    batch_shape: torch.Size | None = None,
) -> Kernel:
    """Create a GPyTorch covariance module from kernel specification."""
    if batch_shape is None:
        batch_shape = torch.Size()

    if isinstance(kernel, Kernel):
        return kernel.to(device=device, dtype=dtype)

    if callable(kernel):
        return _CallableKernel(kernel, length_scale, signal_variance).to(
            device=device, dtype=dtype
        )

    if kernel == "rbf":
        base = RBFKernel(batch_shape=batch_shape).to(device=device, dtype=dtype)
        covar = ScaleKernel(base, batch_shape=batch_shape).to(device=device, dtype=dtype)
        with torch.no_grad():
            covar.base_kernel.lengthscale = torch.tensor(length_scale, dtype=dtype, device=device)
            covar.outputscale = torch.tensor(signal_variance, dtype=dtype, device=device)
        return covar

    if kernel == "matern32":
        base = MaternKernel(nu=1.5, batch_shape=batch_shape).to(device=device, dtype=dtype)
        covar = ScaleKernel(base, batch_shape=batch_shape).to(device=device, dtype=dtype)
        with torch.no_grad():
            covar.base_kernel.lengthscale = torch.tensor(length_scale, dtype=dtype, device=device)
            covar.outputscale = torch.tensor(signal_variance, dtype=dtype, device=device)
        return covar

    if kernel == "linear":
        base = LinearKernel(batch_shape=batch_shape).to(device=device, dtype=dtype)
        covar = ScaleKernel(base, batch_shape=batch_shape).to(device=device, dtype=dtype)
        with torch.no_grad():
            covar.outputscale = torch.tensor(signal_variance, dtype=dtype, device=device)
        return covar

    raise ValueError("Unsupported kernel. Use 'rbf', 'matern32', 'linear', Kernel, or callable.")


def _auto_length_scale(time_index: np.ndarray) -> float:
    """Infer a stable default length-scale from observed time spacing."""
    t_sorted = np.sort(np.asarray(time_index, dtype=float).reshape(-1))
    if t_sorted.size < 2:
        return 1.0
    gaps = np.diff(t_sorted)
    positive_gaps = gaps[gaps > 0]
    if positive_gaps.size == 0:
        return 1.0
    return float(np.median(positive_gaps))


def _normalize_time_axis(t_obs: np.ndarray, t_target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalize time to [0, 1] using observed-time range for stable GP fitting."""
    t_min = float(np.min(t_obs))
    t_max = float(np.max(t_obs))
    span = t_max - t_min
    if span <= 0:
        return np.zeros_like(t_obs, dtype=float), np.zeros_like(t_target, dtype=float)
    return (t_obs - t_min) / span, (t_target - t_min) / span


def bayesian_infer_matrix_over_time(
    observations: np.ndarray,
    time_index: np.ndarray,
    target_time_index: np.ndarray | float | None = None,
    kernel: KernelType = "rbf",
    prior_mean: float = 0.0,
    prior_std: float = 1.0,
    credible_mass: float = 0.95,
    jitter: float = 1e-8,
) -> dict[str, np.ndarray]:
    """BoTorch GP Bayesian inference on all matrix elements over time.

    Parameters are intentionally kept consistent with the original API.
    """
    if prior_std <= 0:
        raise ValueError("prior_std must be > 0")
    if jitter < 0:
        raise ValueError("jitter must be >= 0")
    if not 0 < credible_mass < 1:
        raise ValueError("credible_mass must be in (0, 1)")

    obs = _validate_square_matrices(observations)
    m, n, _ = obs.shape
    t_obs = _validate_time_index(time_index, m)

    if target_time_index is None:
        t_target = t_obs
    else:
        t_target = np.asarray(target_time_index, dtype=float).reshape(-1)

    # Normalize time to reduce optimizer pathologies that can lead to flat posteriors.
    t_obs_model, t_target_model = _normalize_time_axis(t_obs, t_target)

    device = torch.device("cpu")
    dtype = torch.double

    # Keep user-facing API minimal: infer GP hyperparameters from prior and time index.
    length_scale = _auto_length_scale(t_obs_model)
    length_scale = max(1e-2, min(length_scale, 1.0))
    signal_variance = prior_std**2
    obs_std = max(1e-6, 0.1 * prior_std)

    train_x = torch.as_tensor(t_obs_model, dtype=dtype, device=device).unsqueeze(-1)
    test_x = torch.as_tensor(t_target_model, dtype=dtype, device=device).unsqueeze(-1)
    t_count = test_x.shape[0]

    # Fit all n*n matrix elements as independent GP outputs in one batched model.
    # For complex observations, fit real/imag channels as separate outputs and
    # reconstruct complex posterior afterward.
    y_flat = obs.reshape(m, n * n)
    is_complex = np.iscomplexobj(y_flat)
    if is_complex:
        y_train_np = np.concatenate([np.real(y_flat), np.imag(y_flat)], axis=1)
    else:
        y_train_np = np.real(y_flat)

    output_dim = y_train_np.shape[1]
    batch_shape = torch.Size([output_dim])
    # Build explicit batched training tensors: each output channel has its own GP.
    train_x_b = train_x.unsqueeze(0).expand(output_dim, -1, -1)  # (output_dim, m, 1)
    train_y_b = (
        torch.as_tensor(y_train_np.T, dtype=dtype, device=device).unsqueeze(-1)
    )  # (output_dim, m, 1)
    test_x_b = test_x.unsqueeze(0).expand(output_dim, -1, -1)  # (output_dim, t, 1)

    z = (
        torch.distributions.Normal(
            torch.tensor(0.0, dtype=dtype, device=device),
            torch.tensor(1.0, dtype=dtype, device=device),
        )
        .icdf(torch.tensor(0.5 + credible_mass / 2.0, dtype=dtype, device=device))
        .item()
    )

    covar_module = _build_covariance_module(
        kernel=kernel,
        length_scale=length_scale,
        signal_variance=signal_variance,
        dtype=dtype,
        device=device,
        batch_shape=batch_shape,
    )
    mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape).to(
        device=device, dtype=dtype
    )
    with torch.no_grad():
        mean_module.constant.fill_(prior_mean)

    model = SingleTaskGP(
        train_X=train_x_b,
        train_Y=train_y_b,
        covar_module=covar_module,
        mean_module=mean_module,
        input_transform=None,
        outcome_transform=None,
    )
    with torch.no_grad():
        # Initialize (not fix) observation noise; MLL fitting will update it.
        model.likelihood.noise = torch.full_like(model.likelihood.noise, 0.05 * prior_std**2)

    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    model.train()
    model.likelihood.train()
    with gpytorch.settings.cholesky_jitter(jitter):
        fit_gpytorch_mll(mll)

    model.eval()
    model.likelihood.eval()
    with torch.no_grad(), gpytorch.settings.cholesky_jitter(jitter):
        posterior = model.posterior(test_x_b)
        # batched output: (output_dim, t, 1)
        mean_fit = posterior.mean.squeeze(-1).detach().cpu().numpy()  # (output_dim, t)
        var_fit = posterior.variance.squeeze(-1).clamp_min(0.0).detach().cpu().numpy()  # (output_dim, t)
        mean_fit = mean_fit.T  # (t, output_dim)
        var_fit = var_fit.T  # (t, output_dim)

        if is_complex:
            d = n * n
            mean_real = mean_fit[:, :d]
            mean_imag = mean_fit[:, d:]
            var_real = var_fit[:, :d]
            var_imag = var_fit[:, d:]

            posterior_mean = (mean_real + 1j * mean_imag).reshape(test_x.shape[0], n, n)
            # Real-valued total variance proxy for complex entry magnitude.
            posterior_variance = (var_real + var_imag).reshape(test_x.shape[0], n, n)
            std_real = np.sqrt(np.clip(var_real, a_min=0.0, a_max=None))
            std_imag = np.sqrt(np.clip(var_imag, a_min=0.0, a_max=None))
            ci_lower = (
                (mean_real - z * std_real) + 1j * (mean_imag - z * std_imag)
            ).reshape(test_x.shape[0], n, n)
            ci_upper = (
                (mean_real + z * std_real) + 1j * (mean_imag + z * std_imag)
            ).reshape(test_x.shape[0], n, n)
        else:
            posterior_mean = mean_fit.reshape(test_x.shape[0], n, n)
            posterior_variance = var_fit.reshape(test_x.shape[0], n, n)
            posterior_std = np.sqrt(np.clip(posterior_variance, a_min=0.0, a_max=None))
            ci_lower = posterior_mean - z * posterior_std
            ci_upper = posterior_mean + z * posterior_std

        posterior_time_cov = None
        try:
            cov_all = posterior.mvn.covariance_matrix.detach().cpu().numpy()
            # For batched posterior this is (output_dim, t, t)
            if cov_all.ndim == 3:
                posterior_time_cov = cov_all[0]
            else:
                posterior_time_cov = cov_all
        except Exception:
            posterior_time_cov = None

    squeeze = t_count == 1
    results: dict[str, np.ndarray] = {
        "posterior_mean": posterior_mean[0] if squeeze else posterior_mean,
        "posterior_variance": posterior_variance[0] if squeeze else posterior_variance,
        "ci_lower": ci_lower[0] if squeeze else ci_lower,
        "ci_upper": ci_upper[0] if squeeze else ci_upper,
        "posterior_time_cov": posterior_time_cov,
    }
    return results


def run_bayesian_matrix_inference(
    observations: np.ndarray,
    time_index: np.ndarray,
    target_time_index: np.ndarray | float | None = None,
    kernel: KernelType = "rbf",
    prior_mean: float = 0.0,
    prior_std: float = 1.0,
    credible_mass: float = 0.95,
    jitter: float = 1e-8,
) -> dict[str, np.ndarray]:
    """Convenience wrapper with the same signature as the original API."""
    return bayesian_infer_matrix_over_time(
        observations=observations,
        time_index=time_index,
        target_time_index=target_time_index,
        kernel=kernel,
        prior_mean=prior_mean,
        prior_std=prior_std,
        credible_mass=credible_mass,
        jitter=jitter,
    )


def infer_observable_from_shadow_with_botorch(
    observations: np.ndarray,
    operator: np.ndarray,
    time_index: np.ndarray,
    target_time_index: np.ndarray | float | None = None,
    kernel: KernelType = "rbf",
    prior_mean: float = 0.0,
    prior_std: float = 1.0,
    credible_mass: float = 0.68,
    jitter: float = 1e-8,
) -> dict[str, np.ndarray]:
    """Infer an observable at target times from classical-shadow matrices.

    Workflow:
    1) Use classical-shadow matrices (``observations``) to estimate
       ``<O>(t) = Tr(O rho_t)`` at each observed time index.
    2) Run BoTorch GP Bayesian regression on this scalar time series.

    Parameters
    ----------
    observations
        Classical shadow matrices from
        ``construct_classical_shadow_matrices_by_time``.
        Shape: (m, d, d).
    operator
        Observable matrix ``O`` (numpy array-like or object exposing ``.full()``).
        Shape: (d, d).
    time_index
        Observed time points with length ``m``.
    target_time_index
        Target time point(s) for posterior inference. If ``None``, infer on
        observed times.
    kernel, prior_mean, prior_std, credible_mass, jitter
        GP settings passed into BoTorch regression.

    Returns
    -------
    dict
        Keys:
        - ``observable_estimates``: direct shadow-based estimates at observed times.
        - ``posterior_mean`` / ``posterior_variance`` / ``ci_lower`` / ``ci_upper``:
          observable posterior at target times.
        - ``bayesian_results``: full matrix-style output from
          ``bayesian_infer_matrix_over_time`` for the internal 1x1 process.
    """
    obs = np.asarray(observations)
    if obs.ndim != 3 or obs.shape[1] != obs.shape[2]:
        raise ValueError("observations must have shape (m, d, d) with square matrices.")
    m, d, _ = obs.shape

    op_like = operator.full() if hasattr(operator, "full") else operator
    op = np.asarray(op_like)
    if op.shape != (d, d):
        raise ValueError(
            f"operator shape {op.shape} does not match observation matrix shape {(d, d)}."
        )

    t_obs = _validate_time_index(time_index, m)

    # Estimate <O>(t) = Tr(O rho_t) from classical-shadow matrices.
    observable_estimates = np.asarray(
        [np.trace(op @ obs_t) for obs_t in obs],
        dtype=complex if (np.iscomplexobj(op) or np.iscomplexobj(obs)) else float,
    )

    gp_input = observable_estimates.reshape(m, 1, 1)
    gp_results = bayesian_infer_matrix_over_time(
        observations=gp_input,
        time_index=t_obs,
        target_time_index=target_time_index,
        kernel=kernel,
        prior_mean=prior_mean,
        prior_std=prior_std,
        credible_mass=credible_mass,
        jitter=jitter,
    )

    posterior_mean_raw = np.asarray(gp_results["posterior_mean"])
    posterior_variance_raw = np.asarray(gp_results["posterior_variance"])
    ci_lower_raw = np.asarray(gp_results["ci_lower"])
    ci_upper_raw = np.asarray(gp_results["ci_upper"])

    squeeze = posterior_mean_raw.ndim == 2
    posterior_mean = posterior_mean_raw[0, 0] if squeeze else posterior_mean_raw[:, 0, 0]
    posterior_variance = (
        posterior_variance_raw[0, 0] if squeeze else posterior_variance_raw[:, 0, 0]
    )
    ci_lower = ci_lower_raw[0, 0] if squeeze else ci_lower_raw[:, 0, 0]
    ci_upper = ci_upper_raw[0, 0] if squeeze else ci_upper_raw[:, 0, 0]

    return {
        "observable_estimates": observable_estimates,
        "posterior_mean": posterior_mean,
        "posterior_variance": posterior_variance,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "bayesian_results": gp_results,
    }
