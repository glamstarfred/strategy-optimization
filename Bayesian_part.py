#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 21 14:26:53 2026

@author: haiyue
"""

import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from gpytorch.kernels import RBFKernel, ScaleKernel, MaternKernel, PolynomialKernel
from gpytorch.means import ZeroMean, ConstantMean
from gpytorch.distributions import MultivariateNormal
from botorch.models.gpytorch import GPyTorchModel
from torch.distributions import Bernoulli, Normal


class GPBinaryVI(gpytorch.models.ApproximateGP):
    """
    Sparse Variational Gaussian Process model for binary classification.

    This model defines a latent Gaussian process f(x) with:

        f(x) ~ GP(m(x), k(x, x'))

    where:
        - m(x) is a constant mean function,
        - k(x, x') is a scaled Matérn kernel (ν = 0.5),
        - inference is performed using sparse variational inference
          with inducing points.

    Variational Inference
    ---------------------
    The model uses:

        q(u) = N(m, S)

    where u = f(Z) are function values at inducing points Z.
    The Cholesky parameterization ensures S ≽ 0.

    The VariationalStrategy constructs the approximate posterior:

        q(f) = ∫ p(f | u) q(u) du

    and allows inducing locations to be optimized jointly with
    kernel hyperparameters.

    Parameters
    ----------
    inducing_points : torch.Tensor
        Tensor of shape (M, D) specifying the initial inducing inputs.
        M is the number of inducing points, D the input dimension.

    Attributes
    ----------
    mean_module : gpytorch.means.ConstantMean
        Constant prior mean function.

    covar_module : gpytorch.kernels.ScaleKernel
        Kernel of the form:
            σ_f^2 * MaternKernel(ν=0.5)

    Notes
    -----
    - Designed for use with BernoulliLikelihood for binary classification.
    - Inherits from gpytorch.models.ApproximateGP.
    - Inducing locations are learned during training.
    """
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )
        super().__init__(variational_strategy)

        # Mean and kernel
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = ScaleKernel(MaternKernel(nu=0.5))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_gp_classifier(train_x, train_y, num_inducing=50, training_iter=200, lr=0.05):
    """
    Train a sparse variational GP classifier using ELBO maximization.

    This function constructs a GPBinaryVI model with a Bernoulli likelihood
    and optimizes the variational Evidence Lower Bound (ELBO):

        ELBO = E_q(f)[ log p(y | f) ] - KL[q(u) || p(u)]

    where:
        - q(u) is the variational distribution over inducing variables,
        - p(y | f) is the Bernoulli likelihood.

    Parameters
    ----------
    train_x : torch.Tensor
        Training inputs of shape (N, D).

    train_y : torch.Tensor
        Binary targets of shape (N,).
        Expected to be in {0, 1}.

    num_inducing : int, optional (default=50)
        Number of inducing points used for sparse approximation.

    training_iter : int, optional (default=200)
        Number of gradient optimization steps.

    lr : float, optional (default=0.05)
        Learning rate for Adam optimizer.

    Returns
    -------
    model : GPBinaryVI
        Trained variational GP model.

    likelihood : gpytorch.likelihoods.BernoulliLikelihood
        Trained likelihood module.

    Notes
    -----
    - Inducing points are initialized by subsampling train_x.
    - Optimization is performed using Adam.
    - Objective is the negative VariationalELBO.
    - Complexity per iteration is O(NM^2) for M inducing points.
    """
    # Allow either {0,1} labels or {-1,+1} labels from Pauli outcomes.
    train_y = train_y.clone().float()
    unique_y = set(train_y.unique().tolist())
    if unique_y.issubset({-1.0, 1.0}):
        train_y = ((train_y + 1.0) / 2.0).float()
    elif not unique_y.issubset({0.0, 1.0}):
        raise ValueError(
            "train_y must be binary in {0,1} or {-1,+1} for Bernoulli likelihood."
        )

    inducing_points = train_x[:: max(1, len(train_x)//num_inducing)].clone()
    model = GPBinaryVI(inducing_points)
    likelihood = gpytorch.likelihoods.BernoulliLikelihood()
    # model = GPBinaryModel(train_x, train_y, likelihood)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))
    for i in range(training_iter):
        optimizer.zero_grad()

        output = model(train_x)
        loss = -mll(output, train_y)

        loss.backward()
        optimizer.step()

    return model, likelihood

def predict_gp_classifier(model, likelihood, test_x):
    """
    Perform posterior prediction with a trained GP classifier.

    Computes both latent function posterior statistics and
    predictive class probabilities.

    Given a trained model, the posterior over latent function values is:

        q(f_* | X, y, x_*)

    The Bernoulli likelihood maps latent values to probabilities:

        p(y=1 | x_*) = E_q(f_*)[ σ(f_*) ]

    where σ(·) is the inverse link function (e.g., probit or logistic,
    depending on the likelihood configuration).

    Parameters
    ----------
    model : GPBinaryVI
        Trained GP classifier.

    likelihood : gpytorch.likelihoods.BernoulliLikelihood
        Corresponding trained likelihood.

    test_x : torch.Tensor
        Test inputs of shape (N_test, D).

    Returns
    -------
    prob : torch.Tensor
        Predictive mean probability p(y=1 | x_*),
        shape (N_test,).

    latent_mean : torch.Tensor
        Posterior mean of latent function f(x_*),
        shape (N_test,).

    latent_var : torch.Tensor
        Posterior variance of latent function f(x_*),
        shape (N_test,).

    Notes
    -----
    - Model and likelihood are set to evaluation mode.
    - Gradients are disabled via torch.no_grad().
    - `prob` is the mean of the predictive Bernoulli distribution.
    - For calibrated uncertainty analysis, use the full
      predictive distribution rather than only the mean.
    """

    model.eval()
    likelihood.eval()

    with torch.no_grad():
        latent_dist = model(test_x)
        predictive_dist = likelihood(latent_dist)

        prob = predictive_dist.mean          # p(y=1|x)
        latent_mean = latent_dist.mean
        latent_var = latent_dist.variance

    return prob, latent_mean, latent_var


def build_conditional_training_data(measurement_df, qubit, shadow_id):
    """
    Build training tensors for conditional model:
        P(outcome_q = +1 | time, outcome_0, ..., outcome_{q-1}, shadow_id)

    For qubit 0, the feature is only time.
    For qubit q>0, features are [time, previous measured outcomes].
    """
    if qubit < 0:
        raise ValueError("qubit must be non-negative.")

    shadow_block = measurement_df.loc[
        measurement_df["shadow_id"] == shadow_id, ["time", "qubit", "outcome"]
    ]
    if shadow_block.empty:
        raise ValueError(f"No data found for shadow_id={shadow_id}.")

    pivot = shadow_block.pivot_table(
        index="time", columns="qubit", values="outcome", aggfunc="first"
    )

    required_cols = list(range(qubit + 1))
    missing = [c for c in required_cols if c not in pivot.columns]
    if missing:
        raise ValueError(
            f"Missing qubit outcomes {missing} for shadow_id={shadow_id}."
        )

    block = pivot[required_cols].dropna().sort_index()
    times = block.index.to_numpy(dtype=np.float32).reshape(-1, 1)
    if qubit == 0:
        x = times
    else:
        prev_outcomes = block.iloc[:, :qubit].to_numpy(dtype=np.float32)
        x = np.concatenate([times, prev_outcomes], axis=1)

    y = block.iloc[:, qubit].to_numpy(dtype=np.float32)
    train_x = torch.tensor(x, dtype=torch.float32)
    train_y = torch.tensor(y, dtype=torch.float32)
    return train_x, train_y


def train_conditional_gp_classifiers(
    measurement_df,
    qubit_num,
    shadow_ids=None,
    num_inducing=50,
    training_iter=200,
    lr=0.05,
):
    """
    Train one GP classifier per (qubit, shadow_id) with autoregressive inputs.

    Returns
    -------
    model_bank : dict
        Nested dict:
            model_bank[q][s] = {"model": GPBinaryVI, "likelihood": BernoulliLikelihood}
    """
    if shadow_ids is None:
        shadow_ids = sorted(measurement_df["shadow_id"].unique())

    model_bank = {q: {} for q in range(qubit_num)}
    for q in range(qubit_num):
        for s in shadow_ids:
            train_x, train_y = build_conditional_training_data(measurement_df, q, s)
            model, likelihood = train_gp_classifier(
                train_x,
                train_y,
                num_inducing=num_inducing,
                training_iter=training_iter,
                lr=lr,
            )
            model_bank[q][s] = {"model": model, "likelihood": likelihood}
    return model_bank


def predict_conditional_gp_probability(model_entry, time_value, previous_outcomes=None):
    """
    Predict P(outcome=+1) for a single qubit at one time with autoregressive context.
    """
    if previous_outcomes is None:
        previous_outcomes = []

    x_row = np.array([[time_value, *previous_outcomes]], dtype=np.float32)
    test_x = torch.tensor(x_row, dtype=torch.float32)
    prob, _, _ = predict_gp_classifier(
        model_entry["model"], model_entry["likelihood"], test_x
    )
    return float(torch.clamp(prob.squeeze(), 1e-6, 1.0 - 1e-6).item())


def predict_conditional_gp_curve(model_entry, t_values, previous_outcomes=None):
    """
    Predict a full conditional probability curve over time.

    Returns
    -------
    np.ndarray
        Probabilities P(outcome=+1 | t, previous_outcomes), shape (len(t_values),)
    """
    if previous_outcomes is None:
        previous_outcomes = []

    t_values = np.asarray(t_values, dtype=np.float32).reshape(-1, 1)
    n = t_values.shape[0]

    if len(previous_outcomes) == 0:
        x = t_values
    else:
        prev = np.tile(np.asarray(previous_outcomes, dtype=np.float32), (n, 1))
        x = np.concatenate([t_values, prev], axis=1)

    test_x = torch.tensor(x, dtype=torch.float32)
    prob, _, _ = predict_gp_classifier(
        model_entry["model"], model_entry["likelihood"], test_x
    )
    return torch.clamp(prob, 1e-6, 1.0 - 1e-6).detach().cpu().numpy()


def plot_conditional_probability_curves(
    model_bank,
    qubit,
    shadow_id,
    t_values,
    conditioning_patterns=None,
    measurement_df=None,
    ax=None,
):
    """
    Visualize learned conditional probabilities for one (qubit, shadow_id) model.

    Parameters
    ----------
    model_bank : dict
        Output of `train_conditional_gp_classifiers(...)`.
    qubit : int
        Target qubit index.
    shadow_id : int
        Shadow index.
    t_values : array-like
        Time grid used for plotting.
    conditioning_patterns : list[list[int]] or None
        A list of previous-outcome patterns.
        Each pattern length must equal `qubit`.
        If None:
            - qubit=0 -> [[]]
            - otherwise infer from data if `measurement_df` is provided,
              else use all 2^qubit patterns in {-1,+1}.
    measurement_df : pandas.DataFrame or None
        Optional, used only to infer observed conditioning patterns.
    ax : matplotlib.axes.Axes or None
        Optional axis to draw on.

    Returns
    -------
    fig, ax, curves
        `curves` is a dict keyed by tuple(conditioning_pattern) -> np.ndarray
    """
    if qubit not in model_bank or shadow_id not in model_bank[qubit]:
        raise ValueError(f"Missing model for qubit={qubit}, shadow_id={shadow_id}.")

    model_entry = model_bank[qubit][shadow_id]
    t_values = np.asarray(t_values, dtype=float).reshape(-1)

    if conditioning_patterns is None:
        if qubit == 0:
            conditioning_patterns = [[]]
        elif measurement_df is not None:
            block = measurement_df.loc[
                measurement_df["shadow_id"] == shadow_id, ["time", "qubit", "outcome"]
            ]
            pivot = block.pivot_table(
                index="time", columns="qubit", values="outcome", aggfunc="first"
            )
            need = list(range(qubit))
            if all(col in pivot.columns for col in need):
                vals = pivot[need].dropna().to_numpy(dtype=int)
                conditioning_patterns = [list(row) for row in np.unique(vals, axis=0)]
            else:
                conditioning_patterns = [list(p) for p in product([-1, 1], repeat=qubit)]
        else:
            conditioning_patterns = [list(p) for p in product([-1, 1], repeat=qubit)]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
    else:
        fig = ax.figure

    curves = {}
    for pattern in conditioning_patterns:
        if len(pattern) != qubit:
            raise ValueError(
                f"Conditioning pattern length {len(pattern)} must equal qubit={qubit}."
            )
        curve = predict_conditional_gp_curve(
            model_entry, t_values=t_values, previous_outcomes=pattern
        )
        key = tuple(pattern)
        curves[key] = curve

        if qubit == 0:
            label = "no previous qubit"
        else:
            cond_txt = ", ".join([f"m{idx}={val:+d}" for idx, val in enumerate(pattern)])
            label = cond_txt
        ax.plot(t_values, curve, linewidth=2, label=label)

    ax.set_xlabel("time")
    ax.set_ylabel("P(outcome=+1)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"Conditional GP probability (qubit={qubit}, shadow={shadow_id})")
    ax.grid(alpha=0.3)
    if len(conditioning_patterns) > 1 or qubit == 0:
        ax.legend()

    return fig, ax, curves

# In[]
######################### Currently not in use ############################
class MyCustomKernel(gpytorch.kernels.Kernel):
    # if you want to use custom kernel replace what's inside ScaleKernel() with MyCustomKernel()
    has_lengthscale = True  # learnable lengthscale

    def forward(self, x1, x2, diag=False, **params):
        # Pairwise squared Euclidean distance
        diff = (x1.unsqueeze(-2) - x2.unsqueeze(-3)).pow(1).sum(-1)
        K = torch.exp(-0.5 * diff / self.lengthscale**2)  # RBF kernel
        if diag:
            return K.diagonal(dim1=-2, dim2=-1)
        return K

class GPBinaryPrior(gpytorch.models.ExactGP, GPyTorchModel):
    _num_outputs = 1

    def __init__(self, train_x, train_y):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        super().__init__(train_x, train_y, likelihood)

        self.mean_module = ConstantMean()
        self.mean_module.initialize(constant=0.0)
        # MaternKernel(nu=0.5)
        self.covar_module = ScaleKernel(MaternKernel(nu=0.5))
        # self.covar_module = ScaleKernel(MyCustomKernel())

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return MultivariateNormal(mean, covar)


