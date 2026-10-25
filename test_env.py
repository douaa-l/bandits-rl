import numpy as np
from bandit_env import BanditEnv

env = BanditEnv(seed=42)

print("Vraies valeurs q* :", env.q_star.round(2))
print("Meilleure machine :", env.best_action)

# 5 gains de la meilleure machine : ils doivent varier
print("5 gains :", [round(env.pull(env.best_action), 2) for _ in range(5)])

# Moyenne de 1000 gains : elle doit être proche de q* de la meilleure machine
gains = [env.pull(env.best_action) for _ in range(1000)]
print("Moyenne observée sur 1000 essais :", round(np.mean(gains), 2))
print("Vraie valeur q* de cette machine :", round(env.q_star[env.best_action], 2))