"""OpenAI-style evolution strategies (Salimans et al., 2017) in numpy.

Antithetic Gaussian perturbations, centred-rank fitness shaping, Adam on the
estimated gradient, and L2 weight decay. Maximises fitness.

    es = EvolutionStrategy(theta0, population=64, sigma=0.05, lr=0.03, seed=0)
    for _ in range(generations):
        candidates = es.ask()            # (population, n_params)
        es.tell(fitness(candidates))     # one score per candidate
    best = es.theta
"""

from typing import Optional

import numpy as np


def centered_ranks(x: np.ndarray) -> np.ndarray:
    """Ranks scaled to [-0.5, 0.5] (ties broken by order); robust to fitness scale."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 1:
        return np.zeros(1)
    ranks = np.empty(x.size, dtype=np.float64)
    ranks[np.argsort(x, kind="stable")] = np.arange(x.size)
    return ranks / (x.size - 1) - 0.5


class EvolutionStrategy:
    """Antithetic OpenAI-ES with Adam.

    Args:
        theta: Initial parameters (n_params,)
        population: Candidates per generation (even: pairs theta +/- sigma * eps)
        sigma: Perturbation scale
        lr: Adam step size
        weight_decay: L2 coefficient applied to the gradient estimate
        seed: Seed for the perturbations
        mask: Optional (n_params,) 0/1 array; parameters with 0 are never perturbed or
            updated (e.g. a frozen or unused sub-network)
    """

    def __init__(
        self,
        theta: np.ndarray,
        population: int = 64,
        sigma: float = 0.05,
        lr: float = 0.03,
        weight_decay: float = 0.005,
        seed: int = 0,
        beta1: float = 0.9,
        beta2: float = 0.999,
        mask: Optional[np.ndarray] = None,
    ):
        if population < 2 or population % 2:
            raise ValueError("population must be an even number >= 2")
        if sigma <= 0 or lr <= 0:
            raise ValueError("sigma and lr must be positive")
        self.theta = np.asarray(theta, dtype=np.float64).copy()
        self.population = int(population)
        self.sigma = float(sigma)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.beta1, self.beta2 = beta1, beta2
        self.mask = (
            np.ones_like(self.theta)
            if mask is None
            else np.asarray(mask, dtype=np.float64).ravel()
        )
        if self.mask.shape != self.theta.shape:
            raise ValueError("mask must match theta")
        self.rng = np.random.default_rng(seed)
        self.m = np.zeros_like(self.theta)
        self.v = np.zeros_like(self.theta)
        self.t = 0
        self._eps: Optional[np.ndarray] = None

    def ask(self) -> np.ndarray:
        """Candidates (population, n_params): theta + sigma eps rows, then theta - sigma eps."""
        half = self.population // 2
        eps = self.rng.standard_normal((half, self.theta.size)) * self.mask
        self._eps = eps
        return np.concatenate(
            [self.theta + self.sigma * eps, self.theta - self.sigma * eps]
        )

    def tell(self, fitness: np.ndarray) -> np.ndarray:
        """Update theta from the candidates' fitness; returns the gradient estimate."""
        if self._eps is None:
            raise RuntimeError("Call ask() before tell()")
        fitness = np.asarray(fitness, dtype=np.float64).ravel()
        if fitness.size != self.population:
            raise ValueError(f"Expected {self.population} fitness values")
        if not np.all(np.isfinite(fitness)):
            raise ValueError("Fitness contains non-finite values")
        shaped = centered_ranks(fitness)
        half = self.population // 2
        weights = shaped[:half] - shaped[half:]
        grad = weights @ self._eps / (self.population * self.sigma)
        grad = grad - self.weight_decay * self.theta * self.mask
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad * grad
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        step = self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        self.theta = self.theta + step * self.mask
        self._eps = None
        return grad
