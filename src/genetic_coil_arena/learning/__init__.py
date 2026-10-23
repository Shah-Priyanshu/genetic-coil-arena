"""Domain-agnostic learning pieces: batched population MLPs and evolution strategies.

Numpy only. Nothing here imports an environment, ``snake_core`` or numba, so any
domain can train policies with it.

- ``mlp``: a small tanh MLP whose parameters live in one flat vector, evaluated for a
  whole population at once (``(P, n_params)`` parameters, ``(P, M, n_in)`` inputs)
- ``es``: OpenAI-style evolution strategies (antithetic sampling, centred ranks, Adam)
"""

from .es import EvolutionStrategy, centered_ranks
from .mlp import MLP, softmax

__all__ = ["MLP", "EvolutionStrategy", "centered_ranks", "softmax"]
