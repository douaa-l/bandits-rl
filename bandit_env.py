import numpy as np


class BanditEnv:
    """Problème des k machines à sous (slide 13 du cours)."""

    def __init__(self, k=10, seed=None):
        self.k = k
        self.rng = np.random.default_rng(seed)
        # Vraies valeurs cachées q*(a), tirées selon une loi normale N(0, 1)
        self.q_star = self.rng.normal(0, 1, size=k)
        # La meilleure machine (celle que l'agent doit trouver)
        self.best_action = int(np.argmax(self.q_star))

    def pull(self, action):
        """Joue la machine 'action' et renvoie un gain aléatoire N(q*(action), 1)."""
        return self.rng.normal(self.q_star[action], 1)