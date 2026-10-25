import numpy as np


class EpsilonGreedyAgent:
    """Agent epsilon-greedy (slides 11-12), avec mise à jour incrémentale (slides 16-17)."""

    def __init__(self, k=10, epsilon=0.1, q0=0.0, alpha=None, seed=None):
        self.k = k
        self.epsilon = epsilon
        self.alpha = alpha                       # None = moyenne (pas 1/n)
        self.rng = np.random.default_rng(seed)
        self.Q = np.full(k, q0, dtype=float)     # estimations Q(a), valeur initiale q0
        self.N = np.zeros(k, dtype=int)          # nombre de fois où chaque machine a été jouée

    def select_action(self):
        # Avec une probabilité epsilon : explorer (machine au hasard)
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.k))
        # Sinon : exploiter (machine avec le plus grand Q)
        best = np.flatnonzero(self.Q == self.Q.max())
        return int(self.rng.choice(best))

    def update(self, action, reward):
        self.N[action] += 1
        step = 1 / self.N[action] if self.alpha is None else self.alpha
        # Q <- Q + pas * (gain observé - Q)
        self.Q[action] += step * (reward - self.Q[action])