import os
import numpy as np
import matplotlib.pyplot as plt
from agents import EpsilonGreedyAgent, UCBAgent

# Taux de clic réels des 5 titres (inconnus des stratégies)
CTR = np.array([0.040, 0.050, 0.060, 0.035, 0.045])
BEST = int(np.argmax(CTR))
N_VISITORS = 10_000
N_RUNS = 100
os.makedirs("figures", exist_ok=True)


class ClickEnv:
    """Un visiteur voit un titre : clic (1) ou pas de clic (0)."""

    def __init__(self, ctr, seed=None):
        self.ctr = ctr
        self.rng = np.random.default_rng(seed)

    def pull(self, action):
        return float(self.rng.random() < self.ctr[action])


class ABTestAgent:
    """A/B test classique : phase de test à parts égales, puis on garde le gagnant."""

    def __init__(self, k, test_size):
        self.k = k
        self.test_size = test_size
        self.t = 0
        self.views = np.zeros(k)
        self.clicks = np.zeros(k)
        self.winner = None

    def select_action(self):
        self.t += 1
        if self.t <= self.test_size:
            return (self.t - 1) % self.k          # chaque titre à tour de rôle
        if self.winner is None:
            self.winner = int(np.argmax(self.clicks / self.views))
        return self.winner                        # ensuite, toujours le gagnant

    def update(self, action, reward):
        if self.t <= self.test_size:              # on n'apprend que pendant le test
            self.views[action] += 1
            self.clicks[action] += reward


def simulate(make_agent):
    lost = np.zeros(N_VISITORS)
    best_shown = np.zeros(N_VISITORS)
    total_clicks = 0.0
    for run in range(N_RUNS):
        env = ClickEnv(CTR, seed=run)
        agent = make_agent(run)
        for t in range(N_VISITORS):
            action = agent.select_action()
            reward = env.pull(action)
            agent.update(action, reward)
            lost[t] += CTR[BEST] - CTR[action]    # clics perdus en moyenne pour ce visiteur
            best_shown[t] += (action == BEST)
            total_clicks += reward
    return np.cumsum(lost / N_RUNS), best_shown / N_RUNS * 100, total_clicks / N_RUNS


strategies = {
    "A/B test (2000 visiteurs de test)": (lambda run: ABTestAgent(k=5, test_size=2000), "gray"),
    "ε-greedy (ε = 0.1)": (lambda run: EpsilonGreedyAgent(k=5, epsilon=0.1, seed=10_000 + run), "red"),
    "UCB (c = 0.1)": (lambda run: UCBAgent(k=5, c=0.1, seed=10_000 + run), "blue"),
}

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
print(f"Maximum possible : {CTR[BEST] * N_VISITORS:.0f} clics\n")

for name, (make_agent, color) in strategies.items():
    print(f"Simulation en cours : {name}...")
    cum_lost, pct_best, clicks = simulate(make_agent)
    print(f"  clics obtenus : {clicks:.0f} | clics perdus : {cum_lost[-1]:.0f} | "
          f"meilleur titre affiché (1000 derniers visiteurs) : {pct_best[-1000:].mean():.0f} %\n")
    ax1.plot(cum_lost, color=color, label=name)
    ax2.plot(pct_best, color=color, label=name)

ax1.set_ylabel("Clics perdus (cumulés)")
ax1.legend()
ax2.set_ylabel("% de visiteurs qui voient le meilleur titre")
ax2.set_xlabel("Visiteurs")
ax2.set_ylim(0, 100)
ax2.legend()
fig.suptitle(f"Choisir le meilleur titre : A/B test contre bandits ({N_RUNS} simulations)")

plt.tight_layout()
plt.savefig("figures/business_case.png", dpi=150)
plt.show()