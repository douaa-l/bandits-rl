import numpy as np
from bandit_env import BanditEnv
from agents import EpsilonGreedyAgent

env = BanditEnv(seed=42)
n_steps = 1000

for eps in [0, 0.01, 0.1]:
    agent = EpsilonGreedyAgent(k=env.k, epsilon=eps, seed=0)
    rewards = []
    optimal = []

    for t in range(n_steps):
        action = agent.select_action()      # 1. l'agent choisit une machine
        reward = env.pull(action)           # 2. l'environnement donne un gain
        agent.update(action, reward)        # 3. l'agent met à jour son estimation
        rewards.append(reward)
        optimal.append(action == env.best_action)

    print(f"epsilon = {eps} : gain moyen = {np.mean(rewards):.2f}, "
          f"% bons choix = {100 * np.mean(optimal):.1f} %, "
          f"machine préférée = {int(np.argmax(agent.Q))}")

print("Vraie meilleure machine :", env.best_action)