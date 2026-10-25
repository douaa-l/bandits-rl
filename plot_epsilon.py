import os
import matplotlib.pyplot as plt
from agents import EpsilonGreedyAgent
from experiments import run_experiment

N_RUNS = 200      # commence par 200 pour tester, puis passe à 2000
N_STEPS = 1000
os.makedirs("figures", exist_ok=True)

results = {}
for eps in [0, 0.01, 0.1]:
    print(f"Expérience en cours : epsilon = {eps}...")
    results[eps] = run_experiment(
        lambda run: EpsilonGreedyAgent(epsilon=eps, seed=10_000 + run),
        n_runs=N_RUNS, n_steps=N_STEPS,
    )

colors = {0: "green", 0.01: "red", 0.1: "blue"}
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

for eps, (avg_rewards, pct_optimal) in results.items():
    label = "ε = 0 (greedy)" if eps == 0 else f"ε = {eps}"
    ax1.plot(avg_rewards, color=colors[eps], label=label)
    ax2.plot(pct_optimal, color=colors[eps], label=label)

ax1.set_ylabel("Gain moyen")
ax1.legend()
ax2.set_ylabel("% action optimale")
ax2.set_xlabel("Étapes")
ax2.set_ylim(0, 100)
ax2.legend()
fig.suptitle(f"ε-greedy sur un bandit à 10 bras ({N_RUNS} parties)")

plt.tight_layout()
plt.savefig("figures/epsilon_greedy.png", dpi=150)
plt.show()