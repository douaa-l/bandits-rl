import numpy as np
from bandit_env import BanditEnv


def run_experiment(make_agent, n_runs=2000, n_steps=1000, k=10):
    """Joue n_runs parties de n_steps tours et renvoie les moyennes à chaque tour."""
    rewards = np.zeros((n_runs, n_steps))
    optimal = np.zeros((n_runs, n_steps))

    for run in range(n_runs):
        env = BanditEnv(k=k, seed=run)          # un nouveau jeu de machines par partie
        agent = make_agent(run)                 # un nouvel agent, qui ne sait rien

        for t in range(n_steps):
            action = agent.select_action()
            reward = env.pull(action)
            agent.update(action, reward)
            rewards[run, t] = reward
            optimal[run, t] = (action == env.best_action)

    # Moyenne sur toutes les parties, pour chaque tour
    avg_rewards = rewards.mean(axis=0)
    pct_optimal = optimal.mean(axis=0) * 100
    return avg_rewards, pct_optimal