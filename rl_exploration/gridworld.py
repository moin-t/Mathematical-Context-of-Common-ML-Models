"""A finite stochastic grid-world with an explicit transition model."""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class GridWorld:
    """Six-by-six navigation task with hazards and stochastic action slips."""

    rows: int = 6
    cols: int = 6
    slip_probability: float = 0.10
    max_steps: int = 100
    start: tuple = (5, 0)
    goal: tuple = (0, 5)
    hazards: set = field(default_factory=lambda: {(1, 1), (1, 2), (2, 4), (3, 1), (4, 3)})

    def __post_init__(self):
        self.n_states = self.rows * self.cols
        self.n_actions = 4
        self._moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        self.rng = np.random.default_rng()
        self.state = self.to_state(self.start)
        self.steps = 0

    def to_state(self, cell):
        return cell[0] * self.cols + cell[1]

    def to_cell(self, state):
        return divmod(int(state), self.cols)

    @property
    def terminal_states(self):
        return {self.to_state(self.goal), *(self.to_state(x) for x in self.hazards)}

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = self.to_state(self.start)
        self.steps = 0
        return self.state

    def _next_state(self, state, action):
        if state in self.terminal_states:
            return state
        row, col = self.to_cell(state)
        dr, dc = self._moves[action]
        row = min(max(row + dr, 0), self.rows - 1)
        col = min(max(col + dc, 0), self.cols - 1)
        return self.to_state((row, col))

    def reward(self, next_state):
        cell = self.to_cell(next_state)
        if cell == self.goal:
            return 10.0
        if cell in self.hazards:
            return -10.0
        return -0.1

    def transitions(self, state, action):
        """Return (probability, next_state, reward, terminal) outcomes."""
        if state in self.terminal_states:
            return [(1.0, state, 0.0, True)]
        probabilities = np.full(self.n_actions, self.slip_probability / (self.n_actions - 1))
        probabilities[action] = 1.0 - self.slip_probability
        merged = {}
        for actual_action, probability in enumerate(probabilities):
            next_state = self._next_state(state, actual_action)
            merged[next_state] = merged.get(next_state, 0.0) + probability
        return [
            (probability, next_state, self.reward(next_state), next_state in self.terminal_states)
            for next_state, probability in merged.items()
        ]

    def step(self, action):
        outcomes = self.transitions(self.state, action)
        probabilities = [item[0] for item in outcomes]
        index = self.rng.choice(len(outcomes), p=probabilities)
        _, self.state, reward, terminated = outcomes[index]
        self.steps += 1
        truncated = self.steps >= self.max_steps and not terminated
        return self.state, reward, terminated, truncated
