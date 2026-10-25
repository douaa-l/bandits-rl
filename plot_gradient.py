import os
import matplotlib.pyplot as plt
from agents import GradientBanditAgent
from experiments import run_experiment

N_RUNS = 2000
N_STEPS = 1000
TRUE_MEAN = 4     # tous les gains sont autour de +4
os.makedirs("figures", exist_ok=True)

configs = [
    (0.1, True,  "tab:blue",  "α = 0.1, avec référence R̄"),
    (0.4, True,  "lightblue", "α = 0.4, avec référence R̄"),
    (0.1, False, "sienna",    "α = 0.1, sans référence"),
    (0.4, False, "tan",       "α = 0.4, sans référence"),
]

plt.figure(figsize=(8, 5))
for alpha, baseline, color, label in configs:
    print(f"Expérience en cours : {label}...")
    _, pct_optimal = run_experiment(
        lambda run: GradientBanditAgent(alpha=alpha, use_baseline=baseline, seed=10_000 + run),
        n_runs=N_RUNS, n_steps=N_STEPS, true_mean=TRUE_MEAN,
    )
    plt.plot(pct_optimal, color=color, label=label)

plt.xlabel("Étapes")
plt.ylabel("% action optimale")
plt.ylim(0, 100)
plt.legend()
plt.title(f"Gradient bandit : effet de la référence R̄ ({N_RUNS} parties, q* autour de +4)")

plt.tight_layout()
plt.savefig("figures/gradient_bandit.png", dpi=150)
plt.show()