import os
import numpy as np
import matplotlib.pyplot as plt
from agents import EpsilonGreedyAgent, UCBAgent, GradientBanditAgent
from experiments import run_experiment

N_RUNS = 300      # 2000 serait plus précis, mais beaucoup trop long ici
N_STEPS = 1000
os.makedirs("figures", exist_ok=True)

# Pour chaque méthode : (valeurs du paramètre, fonction qui crée l'agent, couleur)
methods = {
    "ε-greedy": (
        [1/128, 1/64, 1/32, 1/16, 1/8, 1/4],
        lambda p, run: EpsilonGreedyAgent(epsilon=p, seed=10_000 + run),
        "red",
    ),
    "gradient bandit": (
        [1/32, 1/16, 1/8, 1/4, 1/2, 1, 2],
        lambda p, run: GradientBanditAgent(alpha=p, seed=10_000 + run),
        "green",
    ),
    "UCB": (
        [1/16, 1/8, 1/4, 1/2, 1, 2, 4],
        lambda p, run: UCBAgent(c=p, seed=10_000 + run),
        "blue",
    ),
    "greedy optimiste (α = 0.1)": (
        [1/4, 1/2, 1, 2, 4],
        lambda p, run: EpsilonGreedyAgent(epsilon=0, q0=p, alpha=0.1, seed=10_000 + run),
        "black",
    ),
}

plt.figure(figsize=(9, 5.5))

for name, (params, make_agent, color) in methods.items():
    scores = []
    for p in params:
        print(f"{name} : paramètre = {p:.4g}...")
        avg_rewards, _ = run_experiment(
            lambda run: make_agent(p, run),
            n_runs=N_RUNS, n_steps=N_STEPS,
        )
        scores.append(avg_rewards.mean())      # gain moyen sur les 1000 tours
    best = int(np.argmax(scores))
    print(f"  -> meilleur réglage : {params[best]:.4g}, gain moyen = {scores[best]:.3f}\n")
    plt.plot(params, scores, color=color, marker="o", label=name)

# Axe horizontal en échelle "doublement", avec des étiquettes 1/128, 1/64, ..., 4
ticks = [2.0 ** i for i in range(-7, 3)]
labels = [f"1/{int(1 / x)}" if x < 1 else str(int(x)) for x in ticks]
plt.xscale("log", base=2)
plt.xticks(ticks, labels)

plt.xlabel("Valeur du paramètre : ε (ε-greedy), α (gradient), c (UCB), Q₀ (optimiste)")
plt.ylabel("Gain moyen sur les 1000 premiers tours")
plt.legend()
plt.title(f"Comparaison des méthodes selon leur paramètre ({N_RUNS} parties par réglage)")

plt.tight_layout()
plt.savefig("figures/comparison.png", dpi=150)
plt.show()