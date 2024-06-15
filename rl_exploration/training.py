"""Tabular Q-learning and greedy-policy evaluation."""

import numpy as np
from .policies import choose_action, random_argmax


DEFAULT_POLICY_CONFIG = {
    "epsilon": 0.10,
    "epsilon_start": 1.0,
    "epsilon_min": 0.02,
    "epsilon_decay": 0.995,
    "temperature_start": 1.0,
    "temperature_min": 0.05,
    "temperature_decay": 0.995,
    "ucb_c": 1.0,
}


def evaluate_greedy_policy(env, q_values, episodes=40, seed=0):
    rng = np.random.default_rng(seed)
    returns, successes = [], []
    for episode in range(episodes):
        state = env.reset(seed=int(rng.integers(2**31 - 1)))
        total = 0.0
        while True:
            action = random_argmax(q_values[state], rng)
            state, reward, terminated, truncated = env.step(action)
            total += reward
            if terminated or truncated:
                returns.append(total)
                successes.append(env.to_cell(state) == env.goal)
                break
    return float(np.mean(returns)), float(np.mean(successes))


def train_q_learning(env, strategy, q_star, episodes=1500, alpha=0.15, gamma=0.95, seed=0, eval_every=25, policy_config=None):
    config = dict(DEFAULT_POLICY_CONFIG)
    if policy_config:
        config.update(policy_config)
    rng = np.random.default_rng(seed)
    q_values = np.zeros((env.n_states, env.n_actions))
    counts = np.zeros_like(q_values, dtype=int)
    history = []
    for episode in range(episodes):
        state = env.reset(seed=int(rng.integers(2**31 - 1)))
        episode_return = 0.0
        while True:
            action = choose_action(strategy, q_values, state, episode, counts, rng, config)
            counts[state, action] += 1
            next_state, reward, terminated, truncated = env.step(action)
            target = reward if terminated else reward + gamma * np.max(q_values[next_state])
            q_values[state, action] += alpha * (target - q_values[state, action])
            episode_return += reward
            state = next_state
            if terminated or truncated:
                break
        if (episode + 1) % eval_every == 0 or episode == 0:
            eval_return, success_rate = evaluate_greedy_policy(env, q_values, seed=seed + episode + 1)
            nonterminal = [s for s in range(env.n_states) if s not in env.terminal_states]
            q_error = float(np.max(np.abs(q_values[nonterminal] - q_star[nonterminal])))
            history.append({
                "episode": episode + 1,
                "training_return": episode_return,
                "evaluation_return": eval_return,
                "success_rate": success_rate,
                "q_error": q_error,
            })
    return q_values, history
