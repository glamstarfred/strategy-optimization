#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 21 14:26:53 2026

@author: haiyue
"""

import torch
import gpytorch
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




