import pandas as pd
import streamlit as st
from agents import EpsilonGreedyAgent, UCBAgent, GradientBanditAgent
from experiments import run_experiment

st.set_page_config(page_title="Bandits manchots", layout="wide")
st.title("Bandits manchots : comparer les stratégies d'exploration")
st.write(
    "Problème à 10 machines à sous (Sutton & Barto, chapitre 2). "
    "Choisis les méthodes et leurs paramètres dans le menu de gauche, puis lance la simulation."
)


def make_agent(method, param, run):
    seed = 10_000 + run
    if method == "ε-greedy":
        return EpsilonGreedyAgent(epsilon=param, seed=seed)
    if method == "Greedy optimiste":
        return EpsilonGreedyAgent(epsilon=0, q0=param, alpha=0.1, seed=seed)
    if method == "UCB":
        return UCBAgent(c=param, seed=seed)
    return GradientBanditAgent(alpha=param, seed=seed)


@st.cache_data
def run_cached(method, param, n_runs, n_steps):
    return run_experiment(lambda run: make_agent(method, param, run), n_runs=n_runs, n_steps=n_steps)


st.sidebar.header("Expérience")
n_runs = st.sidebar.slider("Nombre de parties", 20, 500, 100, step=20)
n_steps = st.sidebar.slider("Nombre de tours par partie", 100, 2000, 1000, step=100)

st.sidebar.header("Méthodes à comparer")
selected = []
if st.sidebar.checkbox("ε-greedy", value=True):
    eps = st.sidebar.slider("ε (probabilité d'explorer)", 0.0, 0.5, 0.1, step=0.01)
    selected.append(("ε-greedy", eps, f"ε-greedy (ε = {eps})"))
if st.sidebar.checkbox("Greedy optimiste"):
    q0 = st.sidebar.slider("Q₀ (valeur initiale)", 0.0, 10.0, 5.0, step=0.5)
    selected.append(("Greedy optimiste", q0, f"Greedy optimiste (Q₀ = {q0})"))
if st.sidebar.checkbox("UCB", value=True):
    c = st.sidebar.slider("c (degré d'exploration)", 0.0, 4.0, 2.0, step=0.1)
    selected.append(("UCB", c, f"UCB (c = {c})"))
if st.sidebar.checkbox("Gradient bandit"):
    alpha = st.sidebar.slider("α (pas d'apprentissage)", 0.01, 1.0, 0.1, step=0.01)
    selected.append(("Gradient bandit", alpha, f"Gradient bandit (α = {alpha})"))

if not selected:
    st.info("Coche au moins une méthode dans le menu de gauche.")
elif st.button("Lancer la simulation", type="primary"):
    rewards, optimal, summary = {}, {}, []
    with st.spinner("Simulation en cours..."):
        for method, param, label in selected:
            avg_rewards, pct_optimal = run_cached(method, param, n_runs, n_steps)
            rewards[label] = avg_rewards
            optimal[label] = pct_optimal
            summary.append({
                "Méthode": label,
                "Gain moyen": round(float(avg_rewards.mean()), 3),
                "% action optimale (fin)": round(float(pct_optimal[-100:].mean()), 1),
            })

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Gain moyen à chaque tour")
        st.line_chart(pd.DataFrame(rewards))
    with col2:
        st.subheader("% de fois où la meilleure machine est choisie")
        st.line_chart(pd.DataFrame(optimal))

    st.subheader("Résumé")
    st.dataframe(pd.DataFrame(summary), hide_index=True)