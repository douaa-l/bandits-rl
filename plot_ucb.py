import os
import matplotlib.pyplot as plt
from agents import EpsilonGreedyAgent, UCBAgent
from experiments import run_experiment

N_RUNS = 2000
N_STEPS = 1000
os.makedirs("figures", exist_ok=True)

print("Expérience en cours : UCB (c = 2)...")
rewards_ucb, _ = run_experiment(
    lambda run: UCBAgent(c=2, seed=10_000 + run),
    n_runs=N_RUNS, n_steps=N_STEPS,
)

print("Expérience en cours : ε-greedy (ε = 0.1)...")
rewards_eps, _ = run_experiment(
    lambda run: EpsilonGreedyAgent(epsilon=0.1, seed=10_000 + run),
    n_runs=N_RUNS, n_steps=N_STEPS,
)

plt.figure(figsize=(8, 5))
plt.plot(rewards_ucb, color="tab:blue", label="UCB (c = 2)")
plt.plot(rewards_eps, color="gray", label="ε-greedy (ε = 0.1)")
plt.xlabel("Étapes")
plt.ylabel("Gain moyen")
plt.legend()
plt.title(f"UCB contre ε-greedy ({N_RUNS} parties)")

plt.tight_layout()
plt.savefig("figures/ucb.png", dpi=150)
plt.show()