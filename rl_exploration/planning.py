"""Exact dynamic programming utilities for finite MDPs."""

import numpy as np


def value_iteration(env, gamma=0.95, tolerance=1e-12, max_iterations=100_000):
    """Compute optimal state and action values using Bellman iteration."""
    values = np.zeros(env.n_states)
    for _ in range(max_iterations):
        new_values = np.zeros_like(values)
        for state in range(env.n_states):
            if state in env.terminal_states:
                continue
            action_values = []
            for action in range(env.n_actions):
                total = sum(
                    probability * (reward + gamma * (not terminal) * values[next_state])
                    for probability, next_state, reward, terminal in env.transitions(state, action)
                )
                action_values.append(total)
            new_values[state] = max(action_values)
        if np.max(np.abs(new_values - values)) < tolerance:
            values = new_values
            break
        values = new_values

    q_star = np.zeros((env.n_states, env.n_actions))
    for state in range(env.n_states):
        if state in env.terminal_states:
            continue
        for action in range(env.n_actions):
            q_star[state, action] = sum(
                probability * (reward + gamma * (not terminal) * values[next_state])
                for probability, next_state, reward, terminal in env.transitions(state, action)
            )
    return values, q_star
