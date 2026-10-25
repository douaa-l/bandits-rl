import os
import matplotlib.pyplot as plt
from agents import EpsilonGreedyAgent
from experiments import run_experiment

N_RUNS = 2000
N_STEPS = 1000
os.makedirs("figures", exist_ok=True)

print("Expérience en cours : optimiste (Q1 = 5, ε = 0)...")
_, pct_optimistic = run_experiment(
    lambda run: EpsilonGreedyAgent(epsilon=0, q0=5, alpha=0.1, seed=10_000 + run),
    n_runs=N_RUNS, n_steps=N_STEPS,
)

print("Expérience en cours : réaliste (Q1 = 0, ε = 0.1)...")
_, pct_realistic = run_experiment(
    lambda run: EpsilonGreedyAgent(epsilon=0.1, q0=0, alpha=0.1, seed=10_000 + run),
    n_runs=N_RUNS, n_steps=N_STEPS,
)

plt.figure(figsize=(8, 5))
plt.plot(pct_optimistic, color="tab:blue", label="Optimiste, greedy (Q1 = 5, ε = 0)")
plt.plot(pct_realistic, color="gray", label="Réaliste, ε-greedy (Q1 = 0, ε = 0.1)")
plt.xlabel("Étapes")
plt.ylabel("% action optimale")
plt.ylim(0, 100)
plt.legend()
plt.title(f"Effet des valeurs initiales optimistes ({N_RUNS} parties)")

plt.tight_layout()
plt.savefig("figures/optimistic_init.png", dpi=150)
plt.show()