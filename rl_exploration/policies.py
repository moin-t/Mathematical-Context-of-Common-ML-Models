"""Exploration policies used by the experiment."""

import numpy as np


def random_argmax(values, rng):
    candidates = np.flatnonzero(np.isclose(values, np.max(values)))
    return int(rng.choice(candidates))


def choose_action(strategy, q_values, state, episode, counts, rng, config):
    if strategy == "epsilon_greedy":
        epsilon = config["epsilon"]
        return int(rng.integers(q_values.shape[1])) if rng.random() < epsilon else random_argmax(q_values[state], rng)
    if strategy == "decaying_epsilon":
        epsilon = max(config["epsilon_min"], config["epsilon_start"] * config["epsilon_decay"] ** episode)
        return int(rng.integers(q_values.shape[1])) if rng.random() < epsilon else random_argmax(q_values[state], rng)
    if strategy == "softmax":
        temperature = max(config["temperature_min"], config["temperature_start"] * config["temperature_decay"] ** episode)
        logits = (q_values[state] - np.max(q_values[state])) / temperature
        probabilities = np.exp(logits) / np.exp(logits).sum()
        return int(rng.choice(q_values.shape[1], p=probabilities))
    if strategy == "ucb":
        untried = np.flatnonzero(counts[state] == 0)
        if len(untried):
            return int(rng.choice(untried))
        total = counts[state].sum()
        bonus = config["ucb_c"] * np.sqrt(np.log(total + 1) / counts[state])
        return random_argmax(q_values[state] + bonus, rng)
    raise ValueError(f"Unknown strategy: {strategy}")
